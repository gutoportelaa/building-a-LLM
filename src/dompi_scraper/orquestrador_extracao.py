#!/usr/bin/env python3
"""
orquestrador_extracao.py — Orquestrador Híbrido (PyMuPDF + Marker Pre-Chunking)
-------------------------------------------------------------------------------
Abordagem Mista (Mix A e B) para extração de PDFs do DOM-PI:
1. Avaliação Leve (PyMuPDF): Mapeia páginas para municípios e calcula score OCR.
2. Fast Path: Se score >= threshold (nativo digital), extrai texto em milissegundos.
3. Slow Path (Pre-Chunking): Se score < threshold, recorta as páginas específicas
   e envia esse mini-PDF para o Marker (GPU).

Isso evita processar PDFs inteiros de 100 páginas no Marker, reduzindo o tempo de
meia hora para alguns segundos nas páginas que realmente importam.

Uso:
    uv run python src/dompi_scraper/orquestrador_extracao.py \
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
        --output-dir dados_brutos_orquestrador \
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
    print("Erro: 'pymupdf' necessário. Instale com: uv add pymupdf")
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
    from dompi_scraper.extrator_marker import (
        create_marker_session,
        extract_with_marker,
    )
    from dompi_scraper.shared_utils import (
        classify_act_type,
        extract_date_from_text,
        compute_content_md5,
    )
except ImportError as e:
    print(f"Erro de importação. Execute a partir da raiz do projeto: {e}")
    sys.exit(1)


# ==============================================================================
# LOGGING
# ==============================================================================

log = logging.getLogger("orquestrador")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    log.setLevel(level)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    log.addHandler(ch)


# ==============================================================================
# FASE 1: MAPEAR PÁGINAS E AVALIAR QUALIDADE
# ==============================================================================

def analisar_e_fatiar_pdf(pdf_path: str, fallback_municipio: str) -> dict:
    """
    Varre o PDF página por página usando PyMuPDF.
    Aplica a regra 'Última Referência' para mapear páginas a municípios.
    Calcula o score OCR médio por município.

    Retorna:
    {
        "nome_municipio": {
            "paginas": [0, 1, ...],
            "scores": [0.8, 0.9, ...],
            "score_medio": 0.85,
            "blocks": [...]  # Blocos ricos (PyMuPDF)
        }
    }
    """
    doc = fitz.open(pdf_path)
    current_city = fallback_municipio
    city_chunks = {}

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = extract_rich_blocks(page)

        # Atualiza a cidade baseando-se nos cabeçalhos da página (Last Known Good)
        for b in blocks:
            cidade_detectada = detect_city_header(b)
            if cidade_detectada:
                current_city = cidade_detectada
                break  # O primeiro cabeçalho válido dita a cidade da página

        # Texto completo da página para calcular o score OCR
        page_text = " ".join([b.get("texto", "") for b in blocks]).strip()
        score = compute_ocr_quality_score(page_text)

        if current_city not in city_chunks:
            city_chunks[current_city] = {
                "paginas": [],
                "scores": [],
                "blocks": [],
            }

        city_chunks[current_city]["paginas"].append(page_num)
        city_chunks[current_city]["scores"].append(score)
        city_chunks[current_city]["blocks"].extend(blocks)

    doc.close()

    # Consolidar médias
    for city, data in city_chunks.items():
        if data["scores"]:
            data["score_medio"] = sum(data["scores"]) / len(data["scores"])
        else:
            data["score_medio"] = 0.0

    return city_chunks


# ==============================================================================
# ORQUESTRAÇÃO PRINCIPAL
# ==============================================================================

def run_orquestrador_pipeline(
    manifest_path: str,
    output_dir: str,
    limite: int,
    threshold: float,
) -> dict:
    if not os.path.exists(manifest_path):
        log.error(f"Manifesto não encontrado: {manifest_path}")
        return {}

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    ok_entries = {fid: e for fid, e in manifest.items() if e.get("status") == "OK"}
    log.info(f"PDFs disponíveis para orquestração: {len(ok_entries)}")

    stats = {
        "total": 0,
        "fast_path": 0,
        "slow_path": 0,
        "erros": 0,
    }

    # Modelos Lazy: Só carregam se o Marker for realmente necessário
    modelos_marker = None
    config_parser = None

    def get_marker_session():
        nonlocal modelos_marker, config_parser
        if modelos_marker is None:
            log.info("🔥 Inicializando Marker (modelos pesados) sob demanda...")
            modelos_marker, config_parser = create_marker_session(force_ocr=False, disable_images=True)
        return modelos_marker, config_parser

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
        # Fase 1: Análise Leve
        chunks = analisar_e_fatiar_pdf(pdf_path, manifest_city)
        log.debug(f"  Analise concluída ({time.time() - t0:.2f}s). Encontrados {len(chunks)} chunks/cidades.")

        for city, data in chunks.items():
            paginas = data["paginas"]
            score = data["score_medio"]
            blocos = data["blocks"]

            log.info(f"  ➤ Cidade: {city} | Páginas: {len(paginas)} | Score Médio: {score:.2f}")

            markdown_text = ""
            extrator_usado = ""
            t_ext = time.time()

            # Fase 2: Roteamento pelo Threshold
            if score >= threshold:
                # FAST PATH: PyMuPDF Markdown
                log.debug("    ⚡ Rota Fast (PyMuPDF)")
                markdown_text = blocks_to_markdown(blocos)
                extrator_usado = "pymupdf"
                stats["fast_path"] += 1
            else:
                # SLOW PATH: Pre-Chunking + Marker
                log.debug("    🐢 Rota Slow (Marker) - Fatiando PDF...")
                mod, conf = get_marker_session()

                # Pre-Chunking: Criar mini-PDF
                doc = fitz.open(pdf_path)
                doc.select(paginas) # Deixa apenas as páginas da cidade
                fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                doc.save(tmp_pdf)
                doc.close()

                # Extração Marker no mini-PDF
                markdown_text = extract_with_marker(mod, conf, tmp_pdf)
                os.remove(tmp_pdf)
                
                extrator_usado = "marker"
                stats["slow_path"] += 1

            t_ext_total = time.time() - t_ext
            
            if not markdown_text or len(markdown_text.strip()) < 50:
                log.warning(f"    ⚠️ Falha ao extrair ({extrator_usado})")
                stats["erros"] += 1
                continue

            # Fase 3: Persistência no Data Lake
            content_hash = compute_content_md5(markdown_text)
            tipo_ato = classify_act_type(markdown_text[:1000], fallback_category="")
            data_pub = extract_date_from_text(markdown_text) or ""

            ano, mes = "sem_ano", "sem_mes"
            if data_pub and len(data_pub) >= 7:
                parts = data_pub.split("-")
                if len(parts) >= 2:
                    ano, mes = parts[0], parts[1]

            frontmatter = generate_frontmatter(
                content_hash=content_hash,
                municipio=city,
                entidade=entidade,
                tipo_ato=tipo_ato,
                data_publicacao=data_pub,
                url_origem=url_origem,
                edicao=edicao,
                sha256_pdf=sha256_pdf,
            )
            full_md = frontmatter + markdown_text

            md_path = build_datalake_path(output_dir, ano, mes, city, f"{content_hash}.md")
            os.makedirs(os.path.dirname(md_path), exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(full_md)

            log.info(f"    ✅ Salvo: {content_hash[:8]}... ({len(markdown_text)} chars | {t_ext_total:.2f}s | {extrator_usado})")
            stats["total"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestrador Híbrido de Extração DOM-PI.")
    parser.add_argument("--manifest", required=True, help="Caminho do manifesto.")
    parser.add_argument("--output-dir", default="dados_brutos_orquestrador", help="Diretório Data Lake.")
    parser.add_argument("--limite", type=int, default=5, help="Limite de PDFs.")
    parser.add_argument("--threshold", type=float, default=0.70, help="Limiar de qualidade (Fast vs Slow).")
    parser.add_argument("--verbose", action="store_true", help="Ativa logs detalhados.")

    args = parser.parse_args()
    _configure_logging(verbose=args.verbose)

    log.info("=" * 60)
    log.info("ORQUESTRADOR HÍBRIDO (PyMuPDF + Marker Pre-Chunking)")
    log.info(f"Threshold: >= {args.threshold} (PyMuPDF) | < {args.threshold} (Marker)")
    log.info("=" * 60)

    t0 = time.time()
    stats = run_orquestrador_pipeline(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limite=args.limite,
        threshold=args.threshold,
    )
    t1 = time.time()

    print("\n" + "=" * 60)
    print("✅  ORQUESTRAÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"  Tempo Total:     {t1 - t0:.1f}s")
    print(f"  Chunks Salvos:   {stats['total']}")
    print(f"  ⚡ Fast Path:     {stats['fast_path']} (PyMuPDF)")
    print(f"  🐢 Slow Path:     {stats['slow_path']} (Marker)")
    print(f"  ❌ Erros:         {stats['erros']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
