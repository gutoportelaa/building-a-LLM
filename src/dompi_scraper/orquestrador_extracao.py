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
    # PaddleOCR e Docling NÃO são importados aqui: torch (cu13/Docling) e
    # paddlepaddle-gpu (cu126) não coexistem no mesmo processo. Cada engine roda
    # num subprocesso isolado (engine_worker) via WorkerClient.
    from dompi_scraper.shared_utils import (
        classify_act_type,
        extract_date_from_text,
        compute_content_md5,
        PALAVRAS_TABELA,
    )
    from dompi_scraper.worker_client import WorkerClient, WorkerError
except ImportError as e:
    print(f"Erro de importação. Execute a partir da raiz do projeto: {e}")
    sys.exit(1)


# ==============================================================================
# LOGGING
# ==============================================================================

log = logging.getLogger("orquestrador")

# Mínimo de caracteres de texto nativo (PyMuPDF) para considerar o documento
# "nativo digital" (não escaneado) e elegível ao dedup pré-extração.
MIN_TEXT_CHARS = 200


def _tz_brasilia():
    """Fuso de Brasília (America/Sao_Paulo) com fallback para UTC-3 fixo."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/Sao_Paulo")
    except Exception:  # noqa: BLE001
        from datetime import timezone, timedelta
        return timezone(timedelta(hours=-3))


class _BrasiliaFormatter(logging.Formatter):
    """Formata o asctime sempre no horário de Brasília, independente do TZ da máquina."""

    _TZ = _tz_brasilia()

    def formatTime(self, record, datefmt=None):
        from datetime import datetime
        dt = datetime.fromtimestamp(record.created, self._TZ)
        return dt.strftime(datefmt or "%H:%M:%S")


def _fmt_dur(segundos: float) -> str:
    """Formata uma duração em h/m/s legível (ex.: '1h03m', '4m12s', '38s')."""
    s = int(max(0, segundos))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _hora_fim(segundos_restantes: float) -> str:
    """Horário de término previsto (HH:MM) no fuso de Brasília."""
    from datetime import datetime, timedelta
    fim = datetime.now(_tz_brasilia()) + timedelta(seconds=max(0, segundos_restantes))
    return fim.strftime("%H:%M")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # %Z anexa "-03"/"BRT" para deixar explícito que é horário de Brasília
    fmt = _BrasiliaFormatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%H:%M:%S BRT")
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
    return contar_gpus() > 0


def contar_gpus() -> int:
    """Número de GPUs CUDA visíveis (via PyTorch). 0 se nenhuma/torch ausente."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def _planejar_gpus(n_gpus: int) -> tuple[str | None, str | None]:
    """
    Decide em qual GPU cada worker roda.

    - >= 2 GPUs: paddle → GPU 0, docling → GPU 1 (isolamento total, usa as 2x L4).
    - 1 GPU:     ambos → GPU 0 (processos separados compartilham a placa sem
                 conflito de runtime, pois estão em venvs distintos).
    - 0 GPU:     ambos em CPU (None).

    Retorna (gpu_paddle, gpu_docling) como strings de índice ou None (CPU).
    """
    if n_gpus >= 2:
        return "0", "1"
    if n_gpus == 1:
        return "0", "0"
    return None, None


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


def _carregar_dla(dla_path: Path) -> set[str]:
    """Carrega o registro de dedup pré-extração (um hash por linha)."""
    if not dla_path.exists():
        return set()
    try:
        with open(dla_path, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    except Exception as e:  # noqa: BLE001
        log.warning(f"Falha ao carregar dedup pré-extração ({e}) — iniciando vazio")
        return set()


def _salvar_dla(dla_path: Path, novos: list[str]) -> None:
    """Anexa (append) os novos hashes de triagem ao registro pré-extração."""
    if not novos:
        return
    with open(dla_path, "a", encoding="utf-8") as f:
        for h in novos:
            f.write(h + "\n")
    log.info(f"Dedup pré-extração: +{len(novos)} hashes → {dla_path}")


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

def _entries_from_dir(pdfs_dir: str) -> dict[str, dict]:
    """
    Constrói entradas no formato do manifesto a partir de uma pasta de PDFs
    (varredura recursiva). Infere município/entidade pela estrutura de caminho
    típica do projeto: territorios/<slug>/pdfs/<municipio>/<entidade>/arquivo.pdf
    """
    import glob
    import hashlib

    entries: dict[str, dict] = {}
    pdfs = sorted(glob.glob(os.path.join(pdfs_dir, "**", "*.pdf"), recursive=True))
    for p in pdfs:
        municipio, entidade = "DESCONHECIDO", ""
        parts = Path(p).parts
        if "pdfs" in parts:
            i = parts.index("pdfs")
            rest = parts[i + 1:-1]  # entre 'pdfs/' e o arquivo
            if len(rest) >= 1:
                municipio = rest[0].replace("_", " ").title()
            if len(rest) >= 2:
                entidade = rest[1].replace("_", " ").title()
        fid = hashlib.md5(p.encode()).hexdigest()
        entries[fid] = {
            "status": "OK", "path": p, "municipio": municipio,
            "entidade": entidade, "url": "", "edicao": "", "sha256": "",
        }
    return entries


def run_orquestrador_pipeline(
    manifest_path: str | None,
    output_dir: str,
    limite: int,
    threshold: float,
    corpus_output: str,
    pdfs_dir: str | None = None,
    python_paddle: str | None = None,
    python_docling: str | None = None,
    gpu_paddle: str | None = "auto",
    gpu_docling: str | None = "auto",
    dpi: int = 200,
    verbose: bool = False,
) -> dict:
    # Origem dos PDFs: manifesto OU pasta (--pdfs-dir)
    if pdfs_dir:
        if not os.path.isdir(pdfs_dir):
            log.error(f"Pasta de PDFs não encontrada: {pdfs_dir}")
            return {}
        ok_entries = _entries_from_dir(pdfs_dir)
        log.info(f"Origem: pasta {pdfs_dir}")
    else:
        if not manifest_path or not os.path.exists(manifest_path):
            log.error(f"Manifesto não encontrado: {manifest_path}")
            return {}
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        ok_entries = {fid: e for fid, e in manifest.items() if e.get("status") == "OK"}
        log.info(f"Origem: manifesto {manifest_path}")

    log.info(f"PDFs disponíveis: {len(ok_entries)}")

    repo_root = Path(__file__).resolve().parents[2]
    n_gpus = contar_gpus()
    gpu = n_gpus > 0

    # Interpretadores dos venvs isolados (ver docs/BENCHMARK_OCR.md §2)
    if not python_paddle:
        python_paddle = str(repo_root / ".venv-paddle" / "bin" / "python")
    if not python_docling:
        python_docling = str(repo_root / ".venv" / "bin" / "python")

    # Planejamento de GPU por worker ("auto" → distribui; senão respeita o pedido)
    auto_paddle, auto_docling = _planejar_gpus(n_gpus)
    if gpu_paddle == "auto":
        gpu_paddle = auto_paddle
    if gpu_docling == "auto":
        gpu_docling = auto_docling

    log.info("=" * 60)
    log.info("Hardware: %d GPU(s) CUDA detectada(s)", n_gpus)
    log.info("Roteamento de GPU → paddle=%s | docling=%s",
             f"GPU{gpu_paddle}" if gpu_paddle not in (None, "") else "CPU",
             f"GPU{gpu_docling}" if gpu_docling not in (None, "") else "CPU")
    log.info("Interpretadores → paddle=%s", python_paddle)
    log.info("                  docling=%s", python_docling)
    log.info("=" * 60)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dedup_path = out_dir / "registro_dedup.ndjson"
    dedup_hashes, dedup_records = _carregar_dedup(dedup_path)

    # Registro de dedup PRÉ-extração (hash do texto da triagem PyMuPDF).
    # Persistido para que re-execuções pulem duplicatas sem custo de motor.
    dla_path = out_dir / "registro_dla_dedup.txt"
    dla_hashes = _carregar_dla(dla_path)
    dla_novos: list[str] = []
    log.info(f"Dedup pré-extração carregado: {len(dla_hashes)} hashes")

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
        "dup_pre":         0,   # duplicatas puladas ANTES de extrair
    }

    # Workers isolados (spawn lazy — só sobem quando a primeira rota os exige)
    _workers: dict[str, WorkerClient] = {}

    def get_paddle_worker() -> WorkerClient | None:
        if "paddle" not in _workers:
            try:
                wc = WorkerClient(
                    engine="paddle", python_exe=python_paddle, repo_root=repo_root,
                    gpu_id=gpu_paddle, dpi=dpi, verbose=verbose,
                )
                wc.start()
                _workers["paddle"] = wc
            except WorkerError as e:
                log.error("Não foi possível iniciar worker PaddleOCR: %s", e)
                _workers["paddle"] = None  # type: ignore[assignment]
        return _workers.get("paddle")

    def get_docling_worker() -> WorkerClient | None:
        if "docling" not in _workers:
            try:
                wc = WorkerClient(
                    engine="docling", python_exe=python_docling, repo_root=repo_root,
                    gpu_id=gpu_docling, dpi=dpi, verbose=verbose,
                )
                wc.start()
                _workers["docling"] = wc
            except WorkerError as e:
                log.error("Não foi possível iniciar worker Docling: %s", e)
                _workers["docling"] = None  # type: ignore[assignment]
        return _workers.get("docling")

    def _extrair_paddle(pdf: str, paginas: list[int]) -> str:
        wc = get_paddle_worker()
        if wc is None:
            return ""
        resp = wc.extract(pdf, paginas)
        if not resp.get("ok"):
            log.warning("    PaddleOCR worker erro: %s", resp.get("error"))
            return ""
        return resp.get("text", "")

    def _extrair_docling(pdf: str, paginas: list[int], ocr: bool = False) -> str:
        wc = get_docling_worker()
        if wc is None:
            return ""
        resp = wc.extract(pdf, paginas, ocr=ocr)
        if not resp.get("ok"):
            log.warning("    Docling worker erro: %s", resp.get("error"))
            return ""
        return resp.get("text", "")

    total_alvo = min(limite, len(ok_entries))
    t_pipeline = time.time()

    # Métricas de ritmo que IGNORAM duplicatas: só contam documentos que
    # realmente passaram por extração (≥1 chunk salvo). Duplicatas são quase
    # instantâneas e distorceriam a média e o ETA.
    t_real = 0.0      # tempo acumulado em PDFs realmente extraídos
    n_real = 0        # nº de PDFs realmente extraídos
    n_dup = 0         # nº de PDFs que foram só duplicata/skip

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

        # Progresso + ETA — média e tempo restante calculados SÓ sobre documentos
        # reais (sem duplicatas). O ETA escala os restantes pela fração observada
        # de documentos reais, então o custo (~zero) das duplicatas não infla a conta.
        feitos = processed - 1
        elapsed_total = time.time() - t_pipeline
        if n_real > 0:
            media = t_real / n_real
            frac_real = n_real / feitos if feitos else 1.0
            restantes_reais = (total_alvo - feitos) * frac_real
            eta = media * restantes_reais
            prog = (f"decorrido {_fmt_dur(elapsed_total)} | "
                    f"média {media:.1f}s/doc (sem dup) | dup {n_dup} | "
                    f"ETA {_fmt_dur(eta)} (~{_hora_fim(eta)})")
        else:
            prog = "estimando ETA..."
        log.info(f"\n[{processed}/{total_alvo}] ({processed/total_alvo*100:.0f}%) "
                 f"{os.path.basename(pdf_path)} | {prog}")

        # marcador para classificar este PDF como real (extraído) ou duplicata
        _saved_antes = stats["total"]
        _pdf_inicio = time.time()

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

            # Texto da triagem (PyMuPDF) — reaproveitado para dedup e roteamento
            texto_dla = " ".join(b.get("texto", "") for b in blocos).strip()
            # "Tem texto nativo utilizável?" decide GPU-route sem pagar OCR de CPU
            tem_texto = (score >= threshold) and (len(texto_dla) >= MIN_TEXT_CHARS)

            log.info(
                f"  → {city} | pgs={len(paginas)} | score={score:.2f} | "
                f"texto={len(texto_dla)}c | nativo={'sim' if tem_texto else 'não'} | "
                f"complexo={'sim' if is_complex else 'não'}"
            )

            # --- Fase 1.5: DEDUP PRÉ-EXTRAÇÃO (evita pagar o motor pesado em duplicatas) ---
            # Só aplica quando há texto nativo confiável (escaneados vão para o
            # dedup pós-extração, já que o texto da triagem é ruído/vazio).
            dla_hash = ""
            if len(texto_dla) >= MIN_TEXT_CHARS:
                dla_hash = compute_content_md5(texto_dla)
                if dla_hash in dla_hashes:
                    log.info(f"    ⏩ Duplicata (pré-extração {dla_hash[:8]}…) — pulando engine")
                    stats["duplicatas"] += 1
                    stats["dup_pre"] += 1
                    continue

            markdown_text  = ""
            extrator_usado = ""
            t_ext = time.time()

            # --- Fase 2: Roteamento ---
            # Princípio: manter o trabalho na GPU e fora do OCR-em-CPU do Docling.
            #   - sem texto nativo (escaneado) → PaddleOCR-GPU (OCR + tabela na GPU)
            #   - nativo + complexo (tabela/fiscal) → Docling do_ocr=False (texto nativo + layout GPU)
            #   - nativo + simples → PyMuPDF (instantâneo, in-process)
            if gpu:
                if not tem_texto:
                    log.debug("    → PaddleOCR CUDA (escaneado)")
                    markdown_text  = _extrair_paddle(pdf_path, paginas)
                    extrator_usado = "paddle-cuda"
                    stats["paddle_cuda"] += 1
                elif is_complex:
                    log.debug("    → Docling CUDA do_ocr=False (nativo+tabela)")
                    markdown_text = _extrair_docling(pdf_path, paginas, ocr=False)
                    if markdown_text:
                        extrator_usado = "docling-cuda"
                        stats["docling_cuda"] += 1
                    else:
                        log.warning("    Docling vazio — fallback PyMuPDF")
                        markdown_text  = blocks_to_markdown(blocos)
                        extrator_usado = "pymupdf-fallback"
                        stats["pymupdf"] += 1
                else:
                    log.debug("    → PyMuPDF fast path (nativo simples)")
                    markdown_text  = blocks_to_markdown(blocos)
                    extrator_usado = "pymupdf"
                    stats["pymupdf"] += 1

            else:  # CPU path
                if not tem_texto:
                    log.debug("    → Tesseract (escaneado, CPU)")
                    markdown_text  = extrair_com_tesseract(pdf_path, paginas)
                    extrator_usado = "tesseract"
                    stats["tesseract"] += 1
                elif is_complex:
                    log.debug("    → PaddleOCR CPU (nativo+tabela)")
                    markdown_text  = _extrair_paddle(pdf_path, paginas)
                    extrator_usado = "paddle-cpu"
                    stats["paddle_cpu"] += 1
                else:
                    log.debug("    → PyMuPDF fast path")
                    markdown_text  = blocks_to_markdown(blocos)
                    extrator_usado = "pymupdf"
                    stats["pymupdf"] += 1

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
                # Aprende o hash de triagem para que a PRÓXIMA execução pule este
                # documento ANTES de extrair (evita re-pagar o motor em retomadas).
                if dla_hash and dla_hash not in dla_hashes:
                    dla_hashes.add(dla_hash)
                    dla_novos.append(dla_hash)
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

            # Registra no dedup (pós-extração) e no dedup pré-extração
            dedup_hashes.add(content_hash)
            dedup_records.append({
                "content_hash":    content_hash,
                "municipio":       city,
                "tipo_ato":        tipo_ato,
                "data_publicacao": data_pub,
                "extrator":        extrator_usado,
            })
            if dla_hash and dla_hash not in dla_hashes:
                dla_hashes.add(dla_hash)
                dla_novos.append(dla_hash)

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

        # Classifica o PDF para as métricas de ritmo: só conta na média/ETA se
        # ao menos um chunk foi REALMENTE extraído e salvo (não-duplicata).
        if stats["total"] > _saved_antes:
            t_real += time.time() - _pdf_inicio
            n_real += 1
        else:
            n_dup += 1

    # Persistência final
    _salvar_dedup(dedup_path, dedup_records)
    _salvar_dla(dla_path, dla_novos)
    if corpus_output:
        _salvar_corpus(Path(corpus_output), corpus_records)

    # Encerramento ordenado dos workers (libera GPU/RAM e loga diagnóstico)
    for nome, wc in _workers.items():
        if wc is not None:
            log.info("Encerrando worker %s...", nome)
            wc.stop()

    return stats


# ==============================================================================
# CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestrador Híbrido DOM-PI (PyMuPDF + PaddleOCR/Docling/Tesseract)."
    )
    parser.add_argument("--manifest",      default=None,                               help="Caminho do manifesto (ou use --pdfs-dir).")
    parser.add_argument("--pdfs-dir",      default=None,                               help="Pasta com PDFs (varredura recursiva). Ex.: territorios/tabuleiros_alto_parnaiba/pdfs")
    parser.add_argument("--output-dir",    default="dados_brutos_orquestrador",        help="Diretório Data Lake.")
    parser.add_argument("--corpus-output", default="corpus_orquestrador.ndjson",       help="Arquivo JSONL Polars de saída.")
    parser.add_argument("--limite",        type=int,   default=10**9,                  help="Limite de PDFs (default: todos).")
    parser.add_argument("--threshold",     type=float, default=0.45,                   help="Score mínimo de texto nativo: acima → nativo (PyMuPDF/Docling); abaixo → escaneado (PaddleOCR-GPU).")
    parser.add_argument("--dpi",           type=int,   default=200,                    help="DPI de rasterização para OCR (workers).")
    parser.add_argument("--python-paddle", default=None,                               help="Interpretador do worker PaddleOCR (default: .venv-paddle).")
    parser.add_argument("--python-docling",default=None,                               help="Interpretador do worker Docling (default: .venv).")
    parser.add_argument("--gpu-paddle",    default="auto",                             help="GPU do worker PaddleOCR: índice (ex '0'), 'cpu' ou 'auto'.")
    parser.add_argument("--gpu-docling",   default="auto",                             help="GPU do worker Docling: índice (ex '1'), 'cpu' ou 'auto'.")
    parser.add_argument("--verbose",       action="store_true",                        help="Logs detalhados (inclui stderr dos workers).")

    args = parser.parse_args()
    if not args.manifest and not args.pdfs_dir:
        parser.error("informe --manifest OU --pdfs-dir")
    _configure_logging(verbose=args.verbose)

    n_gpus = contar_gpus()
    gpu = n_gpus > 0

    # normaliza "cpu" → None para o cliente de worker
    gpu_paddle = None if args.gpu_paddle == "cpu" else args.gpu_paddle
    gpu_docling = None if args.gpu_docling == "cpu" else args.gpu_docling

    log.info("=" * 60)
    log.info("ORQUESTRADOR HÍBRIDO — DOM-PI (workers isolados)")
    log.info(f"Hardware: {n_gpus} GPU(s) CUDA" if gpu else "Hardware: CPU")
    log.info(f"Threshold OCR: {args.threshold} | DPI: {args.dpi}")
    if gpu:
        log.info("Stack: PyMuPDF (DLA) → [worker] PaddleOCR CUDA | [worker] Docling CUDA")
    else:
        log.info("Stack: PyMuPDF → Tesseract | [worker] PaddleOCR CPU")
    log.info("=" * 60)

    t0 = time.time()
    stats = run_orquestrador_pipeline(
        manifest_path=args.manifest,
        pdfs_dir=args.pdfs_dir,
        output_dir=args.output_dir,
        limite=args.limite,
        threshold=args.threshold,
        corpus_output=args.corpus_output,
        python_paddle=args.python_paddle,
        python_docling=args.python_docling,
        gpu_paddle=gpu_paddle,
        gpu_docling=gpu_docling,
        dpi=args.dpi,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("ORQUESTRAÇÃO CONCLUÍDA")
    print("=" * 60)
    total = stats.get('total', 0)
    print(f"  Tempo Total:         {elapsed:.1f}s ({_fmt_dur(elapsed)})")
    if total:
        print(f"  Média:               {elapsed/total:.2f}s/chunk salvo")
    print(f"  Chunks Salvos:       {total}")
    print(f"  PyMuPDF fast path:   {stats.get('pymupdf', 0)}")
    print(f"  PaddleOCR CUDA:      {stats.get('paddle_cuda', 0)}")
    print(f"  Docling CUDA:        {stats.get('docling_cuda', 0)}")
    print(f"  Tesseract:           {stats.get('tesseract', 0)}")
    print(f"  PaddleOCR CPU:       {stats.get('paddle_cpu', 0)}")
    print(f"  Duplicatas:          {stats.get('duplicatas', 0)} "
          f"(puladas pré-extração: {stats.get('dup_pre', 0)})")
    print(f"  Erros:               {stats.get('erros', 0)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
