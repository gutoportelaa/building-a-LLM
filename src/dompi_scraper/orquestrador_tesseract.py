#!/usr/bin/env python3
"""
orquestrador_tesseract.py — Orquestrador Híbrido (PyMuPDF + Tesseract OCR)
-------------------------------------------------------------------------------
Abordagem Mista para extração de PDFs do DOM-PI:
1. Avaliação Leve (PyMuPDF): Mapeia páginas para municípios e calcula score OCR.
2. Fast Path: Se score >= threshold (nativo digital), extrai texto em milissegundos.
3. Slow Path (Tesseract): Se score < threshold, recorta as páginas específicas
   e usa pytesseract para ler a imagem. 

Vantagem: O Tesseract consegue extrair texto de uma página 4-em-1 escaneada
em poucos segundos, enquanto o Marker demoraria 25 minutos. A qualidade será
um pouco inferior, mas o ganho de tempo é astronômico.

Dependências de sistema:
  sudo apt install tesseract-ocr tesseract-ocr-por poppler-utils
  uv add pytesseract pdf2image

Uso:
    uv run python src/dompi_scraper/orquestrador_tesseract.py \
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
        --output-dir dados_brutos_tesseract \
        --limite 3 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Erro: 'pymupdf' necessário.")
    sys.exit(1)

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Erro: 'pytesseract' e 'Pillow' (PIL) necessários. Execute: uv add pytesseract pillow")
    sys.exit(1)

# Importa lógicas dos módulos existentes
try:
    from dompi_scraper.processar_pdfs import (
        extract_rich_blocks,
        detect_city_header,
        compute_ocr_quality_score,
        blocks_to_markdown,
        build_datalake_path,
        generate_frontmatter,
    )
    from dompi_scraper.shared_utils import (
        classify_act_type,
        extract_date_from_text,
        compute_content_md5,
    )
except ImportError as e:
    print(f"Erro de importação: {e}")
    sys.exit(1)

log = logging.getLogger("orq_tesseract")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log.setLevel(level)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    log.addHandler(ch)


import unicodedata
import re

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

def analisar_e_fatiar_pdf(pdf_path: str, fallback_municipio: str) -> dict:
    doc = fitz.open(pdf_path)
    city_chunks = {fallback_municipio: {"paginas": [], "scores": [], "blocks": []}}

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = extract_rich_blocks(page)

        page_text = " ".join([b.get("texto", "") for b in blocks]).strip()
        score = compute_ocr_quality_score(page_text)

        city_chunks[fallback_municipio]["paginas"].append(page_num)
        city_chunks[fallback_municipio]["scores"].append(score)
        city_chunks[fallback_municipio]["blocks"].extend(blocks)

    doc.close()

    for city, data in city_chunks.items():
        data["score_medio"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0

    return city_chunks


def extract_with_tesseract(pdf_path: str) -> str:
    """Extrai texto do PDF usando Tesseract OCR via imagens renderizadas pelo PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            # Renderiza a página com escala ~300 DPI (300/72 = 4.16)
            matrix = fitz.Matrix(4.16, 4.16)
            pix = page.get_pixmap(matrix=matrix)
            
            # Converte para Pillow Image
            if pix.alpha:
                img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples).convert("RGB")
            else:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
            # lang='por' requer o pacote tesseract-ocr-por no sistema
            text = pytesseract.image_to_string(img, lang='por')
            full_text.append(text)
            
        doc.close()
        return "\n\n".join(full_text)
    except Exception as e:
        log.error(f"Erro no Tesseract: {e}")
        return ""


def run_tesseract_pipeline(manifest_path: str, output_dir: str, limite: int, threshold: float) -> dict:
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    ok_entries = {fid: e for fid, e in manifest.items() if e.get("status") == "OK"}
    stats = {"total": 0, "fast_path": 0, "slow_path": 0, "erros": 0}
    processed = 0

    for fid, entry in ok_entries.items():
        if processed >= limite:
            break

        pdf_path = entry.get("path", "")
        if not pdf_path or not os.path.exists(pdf_path):
            continue

        processed += 1
        manifest_city = entry.get("municipio", "DESCONHECIDO")
        entidade = entry.get("entidade", "")
        url_origem = entry.get("url", "")
        edicao = entry.get("edicao", "")
        sha256_pdf = entry.get("sha256", "")

        log.info(f"\n[{processed}/{min(limite, len(ok_entries))}] Processando {os.path.basename(pdf_path)}")
        t0 = time.time()
        
        chunks = analisar_e_fatiar_pdf(pdf_path, manifest_city)
        log.debug(f"  Analise concluída ({time.time() - t0:.2f}s). {len(chunks)} chunks.")

        for city, data in chunks.items():
            paginas = data["paginas"]
            score = data["score_medio"]
            blocos = data["blocks"]

            log.info(f"  ➤ Cidade: {city} | Páginas: {len(paginas)} | Score Médio: {score:.2f}")

            markdown_text = ""
            extrator_usado = ""
            t_ext = time.time()

            if score >= threshold:
                log.debug("    ⚡ Rota Fast (PyMuPDF)")
                markdown_text = blocks_to_markdown(blocos)
                extrator_usado = "pymupdf"
                stats["fast_path"] += 1
            else:
                log.debug("    🐢 Rota Slow (Tesseract) - Fatiando PDF...")
                doc = fitz.open(pdf_path)
                doc.select(paginas)
                fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                doc.save(tmp_pdf)
                doc.close()

                markdown_text = extract_with_tesseract(tmp_pdf)
                os.remove(tmp_pdf)
                
                extrator_usado = "tesseract"
                stats["slow_path"] += 1

            t_ext_total = time.time() - t_ext
            
            if not markdown_text or len(markdown_text.strip()) < 50:
                log.warning(f"    ⚠️ Falha ao extrair ({extrator_usado})")
                stats["erros"] += 1
                continue

            content_hash = compute_content_md5(markdown_text)
            tipo_ato = classify_act_type(markdown_text[:1000], fallback_category="")
            data_pub = extract_date_from_text(markdown_text) or ""

            frontmatter = generate_frontmatter(
                content_hash=content_hash, municipio=manifest_city, entidade=entidade,
                tipo_ato=tipo_ato, data_publicacao=data_pub, url_origem=url_origem,
                edicao=edicao, sha256_pdf=sha256_pdf
            )
            
            city_slug = slugify(manifest_city)
            md_path = os.path.join(output_dir, city_slug, f"{content_hash}.md")
            os.makedirs(os.path.dirname(md_path), exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(frontmatter + markdown_text)

            log.info(f"    ✅ Salvo: {content_hash[:8]}... ({len(markdown_text)} chars | {t_ext_total:.2f}s | {extrator_usado})")
            stats["total"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestrador Híbrido com Tesseract.")
    parser.add_argument("--manifest", required=True, help="Caminho do manifesto.")
    parser.add_argument("--output-dir", default="dados_brutos_tesseract", help="Data Lake.")
    parser.add_argument("--limite", type=int, default=5, help="Limite de PDFs.")
    parser.add_argument("--threshold", type=float, default=0.70, help="Limiar de qualidade.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    _configure_logging(verbose=args.verbose)

    log.info("=" * 60)
    log.info("ORQUESTRADOR HÍBRIDO (PyMuPDF + Tesseract)")
    log.info("=" * 60)

    t0 = time.time()
    stats = run_tesseract_pipeline(args.manifest, args.output_dir, args.limite, args.threshold)
    t1 = time.time()

    print("\n" + "=" * 60)
    print("✅  ORQUESTRAÇÃO TESSERACT CONCLUÍDA")
    print("=" * 60)
    print(f"  Tempo Total:     {t1 - t0:.1f}s")
    print(f"  Chunks Salvos:   {stats['total']}")
    print(f"  ⚡ Fast Path:     {stats['fast_path']} (PyMuPDF)")
    print(f"  🐢 Slow Path:     {stats['slow_path']} (Tesseract)")
    print(f"  ❌ Erros:         {stats['erros']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
