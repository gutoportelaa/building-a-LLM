#!/usr/bin/env python3
"""
orquestrador_extracao.py — Orquestrador Híbrido (PyMuPDF + PaddleOCR/Docling/Tesseract)
----------------------------------------------------------------------------------------
Stack de extração adaptada ao hardware disponível:

  GPU (CUDA):
    1. Triagem DLA via PyMuPDF — mapeia páginas por município, calcula score OCR e
       detecta complexidade de layout (tabelas, keywords fiscais).
    2. Texto simples (score >= threshold, sem tabelas) → PaddleOCR CUDA
    3. Documento complexo (tabelas, valores fiscais)   → Docling via PyTorch/CUDA

  Sem GPU:
    1. PyMuPDF lê metadados e score OCR do documento
    2. Texto nativo digital (score >= threshold)       → PyMuPDF fast path
    3. Escaneado mundano (OCR sem tabelas)             → Tesseract
    4. Complexo (tabelas / keywords fiscais)           → PaddleOCR CPU

  Manipulação de dados: Polars (corpus JSONL, dedup, estatísticas)

Uso:
    uv run python src/dompi_scraper/orquestrador_extracao.py \\
        --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \\
        --output-dir dados_brutos_orquestrador \\
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

try:
    import polars as pl
except ImportError:
    print("Erro: 'polars' necessário. Instale com: uv add polars")
    sys.exit(1)

try:
    import pytesseract
    from PIL import Image as PILImage
    _TESSERACT_OK = True
except ImportError:
    _TESSERACT_OK = False

try:
    from dompi_scraper.processar_pdfs import (
        extract_rich_blocks,
        detect_city_header,
        compute_ocr_quality_score,
        blocks_to_markdown,
        build_datalake_path,
        generate_frontmatter,
    )
    from dompi_scraper.extrator_paddle import (
        criar_engine_paddle,
        extrair_pdf_paddle,
        PALAVRAS_TABELA,
    )
    from dompi_scraper.worker_docling import (
        carregar_docling,
        extrair_com_docling,
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
# DETECÇÃO DE GPU
# ==============================================================================

def detectar_gpu() -> bool:
    """Retorna True se CUDA disponível via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ==============================================================================
# FASE 1: TRIAGEM DLA — MAPEAR PÁGINAS E AVALIAR COMPLEXIDADE
# ==============================================================================

def analisar_e_fatiar_pdf(pdf_path: str, fallback_municipio: str) -> dict:
    """
    Varre o PDF página por página usando PyMuPDF (DLA leve).
    Aplica a regra 'Última Referência' para mapear páginas a municípios.
    Calcula score OCR médio e detecta complexidade de layout por município.

    Retorna:
    {
        "nome_municipio": {
            "paginas":      [0, 1, ...],
            "scores":       [0.8, 0.9, ...],
            "score_medio":  0.85,
            "is_complex":   False,   # tabelas ou keywords fiscais detectados
            "blocks":       [...]
        }
    }
    """
    doc = fitz.open(pdf_path)
    current_city = fallback_municipio
    city_chunks: dict[str, dict] = {}

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = extract_rich_blocks(page)

        for b in blocks:
            cidade_detectada = detect_city_header(b)
            if cidade_detectada:
                current_city = cidade_detectada
                break

        page_text = " ".join(b.get("texto", "") for b in blocks).strip()
        score = compute_ocr_quality_score(page_text)

        if current_city not in city_chunks:
            city_chunks[current_city] = {
                "paginas": [],
                "scores": [],
                "blocks": [],
                "is_complex": False,
            }

        city_chunks[current_city]["paginas"].append(page_num)
        city_chunks[current_city]["scores"].append(score)
        city_chunks[current_city]["blocks"].extend(blocks)

        # Detecção de complexidade por keywords fiscais na página
        if not city_chunks[current_city]["is_complex"]:
            texto_lower = page_text.lower()
            if any(kw in texto_lower for kw in PALAVRAS_TABELA):
                city_chunks[current_city]["is_complex"] = True

    doc.close()

    for city, data in city_chunks.items():
        data["score_medio"] = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0

    return city_chunks


# ==============================================================================
# EXTRATORES
# ==============================================================================

def _criar_mini_pdf(pdf_path: str, paginas: list[int]) -> str:
    """Cria um PDF temporário contendo apenas as páginas especificadas."""
    doc = fitz.open(pdf_path)
    doc.select(paginas)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(tmp_path)
    doc.close()
    return tmp_path


def extrair_com_tesseract(pdf_path: str, paginas: list[int], dpi: int = 200) -> str:
    """Extrai texto via Tesseract (PT-BR) das páginas especificadas."""
    if not _TESSERACT_OK:
        log.error("Tesseract não disponível. Instale: uv add pytesseract pillow")
        return ""

    doc = fitz.open(pdf_path)
    partes: list[str] = []

    for pn in paginas:
        if pn >= len(doc):
            continue
        page = doc.load_page(pn)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            text = pytesseract.image_to_string(img, lang="por", timeout=60)
            if text.strip():
                partes.append(text.strip())
        except Exception as e:
            log.warning(f"Tesseract falhou na página {pn}: {e}")

    doc.close()
    return "\n\n".join(partes)


# ==============================================================================
# PERSISTÊNCIA COM POLARS
# ==============================================================================

_DEDUP_SCHEMA = {
    "content_hash":    pl.Utf8,
    "municipio":       pl.Utf8,
    "tipo_ato":        pl.Utf8,
    "data_publicacao": pl.Utf8,
    "extrator":        pl.Utf8,
}

_CORPUS_SCHEMA = {
    "id_publicacao":   pl.Utf8,
    "municipio":       pl.Utf8,
    "tipo_ato":        pl.Utf8,
    "data_publicacao": pl.Utf8,
    "extrator":        pl.Utf8,
    "texto":           pl.Utf8,
    "n_chars":         pl.Int64,
}


def _carregar_dedup(dedup_path: Path) -> tuple[set[str], list[dict]]:
    """
    Carrega o registro de deduplicação com Polars.
    Retorna (set de hashes conhecidos, lista de dicts para append).
    """
    if not dedup_path.exists():
        return set(), []
    try:
        df = pl.read_ndjson(dedup_path, schema=_DEDUP_SCHEMA)
        hashes = set(df["content_hash"].to_list())
        records = df.to_dicts()
        log.info(f"Dedup carregado: {len(hashes)} hashes")
        return hashes, records
    except Exception as e:
        log.warning(f"Falha ao carregar dedup ({e}) — iniciando vazio")
        return set(), []


def _salvar_dedup(dedup_path: Path, records: list[dict]) -> None:
    """Salva o registro de deduplicação atomicamente via Polars."""
    if not records:
        return
    tmp = str(dedup_path) + ".tmp"
    pl.DataFrame(records, schema=_DEDUP_SCHEMA).write_ndjson(tmp)
    os.replace(tmp, str(dedup_path))
    log.info(f"Dedup salvo: {len(records)} hashes → {dedup_path}")


def _salvar_corpus(corpus_path: Path, corpus_records: list[dict]) -> None:
    """Grava o corpus JSONL (textos extraídos) com Polars."""
    if not corpus_records:
        return
    tmp = str(corpus_path) + ".tmp"
    pl.DataFrame(corpus_records, schema=_CORPUS_SCHEMA).write_ndjson(tmp)
    os.replace(tmp, str(corpus_path))
    log.info(f"Corpus salvo: {len(corpus_records)} registros → {corpus_path}")


# ==============================================================================
# ORQUESTRAÇÃO PRINCIPAL
# ==============================================================================

def run_orquestrador_pipeline(
    manifest_path: str,
    output_dir: str,
    limite: int,
    threshold: float,
    corpus_output: str,
) -> dict:
    if not os.path.exists(manifest_path):
        log.error(f"Manifesto não encontrado: {manifest_path}")
        return {}

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    ok_entries = {fid: e for fid, e in manifest.items() if e.get("status") == "OK"}
    log.info(f"PDFs disponíveis: {len(ok_entries)}")

    gpu = detectar_gpu()
    log.info(f"GPU (CUDA): {'disponível' if gpu else 'não disponível'}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedup_path = out_dir / "registro_dedup.ndjson"
    dedup_hashes, dedup_records = _carregar_dedup(dedup_path)

    corpus_records: list[dict] = []

    stats = {
        "total":           0,
        "pymupdf":         0,
        "paddle_cuda":     0,
        "docling_cuda":    0,
        "tesseract":       0,
        "paddle_cpu":      0,
        "erros":           0,
        "duplicatas":      0,
    }

    # Inicialização lazy dos motores pesados
    _paddle_engine = None
    _docling_conv  = None

    def get_paddle(use_gpu: bool):
        nonlocal _paddle_engine
        if _paddle_engine is None:
            log.info(f"Inicializando PaddleOCR (gpu={use_gpu})...")
            _paddle_engine = criar_engine_paddle(use_gpu=use_gpu)
        return _paddle_engine

    def get_docling():
        nonlocal _docling_conv
        if _docling_conv is None:
            log.info("Inicializando Docling (CUDA)...")
            _docling_conv = carregar_docling()
        return _docling_conv

    processed = 0
    for fid, entry in ok_entries.items():
        if processed >= limite:
            break

        pdf_path = entry.get("path", "")
        if not pdf_path or not os.path.exists(pdf_path):
            continue

        processed += 1
        manifest_city = entry.get("municipio", "DESCONHECIDO")
        entidade      = entry.get("entidade", "")
        url_origem    = entry.get("url", "")
        edicao        = entry.get("edicao", "")
        sha256_pdf    = entry.get("sha256", "")

        log.info(f"\n[{processed}/{min(limite, len(ok_entries))}] {os.path.basename(pdf_path)}")

        t0 = time.time()

        # --- Fase 1: Triagem DLA (PyMuPDF) ---
        try:
            chunks = analisar_e_fatiar_pdf(pdf_path, manifest_city)
        except Exception as e:
            log.error(f"Falha na triagem DLA: {e}")
            stats["erros"] += 1
            continue

        log.debug(f"  DLA: {len(chunks)} chunk(s) em {time.time()-t0:.2f}s")

        for city, data in chunks.items():
            paginas    = data["paginas"]
            score      = data["score_medio"]
            blocos     = data["blocks"]
            is_complex = data["is_complex"]

            log.info(
                f"  → {city} | pgs={len(paginas)} | score={score:.2f} | "
                f"complexo={'sim' if is_complex else 'não'}"
            )

            markdown_text  = ""
            extrator_usado = ""
            t_ext = time.time()

            # --- Fase 2: Roteamento ---
            if gpu:
                if score >= threshold and not is_complex:
                    # GPU + texto simples nativo: PyMuPDF já extraiu tudo
                    markdown_text  = blocks_to_markdown(blocos)
                    extrator_usado = "pymupdf"
                    stats["pymupdf"] += 1
                elif is_complex:
                    # GPU + complexo: Docling CUDA
                    log.debug("    → Docling CUDA")
                    conv = get_docling()
                    if conv is None:
                        log.warning("    Docling indisponível — fallback PaddleOCR CUDA")
                        tmp = _criar_mini_pdf(pdf_path, paginas)
                        try:
                            res = extrair_pdf_paddle(get_paddle(use_gpu=True), tmp)
                            markdown_text = res.texto_completo
                        finally:
                            os.remove(tmp)
                        extrator_usado = "paddle-cuda-fallback"
                        stats["paddle_cuda"] += 1
                    else:
                        markdown_text  = extrair_com_docling(conv, pdf_path, pages=paginas)
                        extrator_usado = "docling-cuda"
                        stats["docling_cuda"] += 1
                else:
                    # GPU + OCR simples: PaddleOCR CUDA
                    log.debug("    → PaddleOCR CUDA")
                    tmp = _criar_mini_pdf(pdf_path, paginas)
                    try:
                        res = extrair_pdf_paddle(get_paddle(use_gpu=True), tmp)
                        markdown_text = res.texto_completo
                    finally:
                        os.remove(tmp)
                    extrator_usado = "paddle-cuda"
                    stats["paddle_cuda"] += 1

            else:  # CPU path
                if score >= threshold:
                    # CPU + digital nativo: PyMuPDF fast path
                    log.debug("    → PyMuPDF fast path")
                    markdown_text  = blocks_to_markdown(blocos)
                    extrator_usado = "pymupdf"
                    stats["pymupdf"] += 1
                elif is_complex:
                    # CPU + complexo: PaddleOCR CPU
                    log.debug("    → PaddleOCR CPU")
                    tmp = _criar_mini_pdf(pdf_path, paginas)
                    try:
                        res = extrair_pdf_paddle(get_paddle(use_gpu=False), tmp)
                        markdown_text = res.texto_completo
                    finally:
                        os.remove(tmp)
                    extrator_usado = "paddle-cpu"
                    stats["paddle_cpu"] += 1
                else:
                    # CPU + mundano: Tesseract
                    log.debug("    → Tesseract")
                    markdown_text  = extrair_com_tesseract(pdf_path, paginas)
                    extrator_usado = "tesseract"
                    stats["tesseract"] += 1

            t_ext_total = time.time() - t_ext

            if not markdown_text or len(markdown_text.strip()) < 50:
                log.warning(f"    Extração vazia ou insuficiente ({extrator_usado})")
                stats["erros"] += 1
                continue

            # --- Fase 3: Deduplicação ---
            content_hash = compute_content_md5(markdown_text)
            if content_hash in dedup_hashes:
                log.debug(f"    Duplicata: {content_hash[:8]}...")
                stats["duplicatas"] += 1
                continue

            # --- Fase 4: Metadados e Persistência ---
            tipo_ato = classify_act_type(markdown_text, fallback_category="")
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
            full_md  = frontmatter + markdown_text
            md_path  = build_datalake_path(output_dir, ano, mes, city, f"{content_hash}.md")
            os.makedirs(os.path.dirname(md_path), exist_ok=True)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(full_md)

            # Registra no dedup
            dedup_hashes.add(content_hash)
            dedup_records.append({
                "content_hash":    content_hash,
                "municipio":       city,
                "tipo_ato":        tipo_ato,
                "data_publicacao": data_pub,
                "extrator":        extrator_usado,
            })

            # Registra no corpus Polars
            corpus_records.append({
                "id_publicacao":   content_hash,
                "municipio":       city,
                "tipo_ato":        tipo_ato,
                "data_publicacao": data_pub,
                "extrator":        extrator_usado,
                "texto":           markdown_text,
                "n_chars":         len(markdown_text),
            })

            log.info(
                f"    Salvo: {content_hash[:8]}... "
                f"({len(markdown_text)} chars | {t_ext_total:.2f}s | {extrator_usado})"
            )
            stats["total"] += 1

    # Persistência final
    _salvar_dedup(dedup_path, dedup_records)
    if corpus_output:
        _salvar_corpus(Path(corpus_output), corpus_records)

    return stats


# ==============================================================================
# CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestrador Híbrido DOM-PI (PyMuPDF + PaddleOCR/Docling/Tesseract)."
    )
    parser.add_argument("--manifest",      required=True,                              help="Caminho do manifesto.")
    parser.add_argument("--output-dir",    default="dados_brutos_orquestrador",        help="Diretório Data Lake.")
    parser.add_argument("--corpus-output", default="corpus_orquestrador.ndjson",       help="Arquivo JSONL Polars de saída.")
    parser.add_argument("--limite",        type=int,   default=5,                      help="Limite de PDFs.")
    parser.add_argument("--threshold",     type=float, default=0.70,                   help="Limiar de qualidade OCR.")
    parser.add_argument("--verbose",       action="store_true",                        help="Logs detalhados.")

    args = parser.parse_args()
    _configure_logging(verbose=args.verbose)

    gpu = detectar_gpu()

    log.info("=" * 60)
    log.info("ORQUESTRADOR HÍBRIDO — DOM-PI")
    log.info(f"Hardware: {'GPU CUDA' if gpu else 'CPU'}")
    log.info(f"Threshold OCR: {args.threshold}")
    if gpu:
        log.info("Stack: PyMuPDF (DLA) → PaddleOCR CUDA | Docling CUDA")
    else:
        log.info("Stack: PyMuPDF → Tesseract | PaddleOCR CPU")
    log.info("=" * 60)

    t0 = time.time()
    stats = run_orquestrador_pipeline(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limite=args.limite,
        threshold=args.threshold,
        corpus_output=args.corpus_output,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("ORQUESTRAÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"  Tempo Total:         {elapsed:.1f}s")
    print(f"  Chunks Salvos:       {stats.get('total', 0)}")
    print(f"  PyMuPDF fast path:   {stats.get('pymupdf', 0)}")
    print(f"  PaddleOCR CUDA:      {stats.get('paddle_cuda', 0)}")
    print(f"  Docling CUDA:        {stats.get('docling_cuda', 0)}")
    print(f"  Tesseract:           {stats.get('tesseract', 0)}")
    print(f"  PaddleOCR CPU:       {stats.get('paddle_cpu', 0)}")
    print(f"  Duplicatas:          {stats.get('duplicatas', 0)}")
    print(f"  Erros:               {stats.get('erros', 0)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
