#!/usr/bin/env python3
"""
engine_worker.py — Worker de extração isolado por engine (PaddleOCR | Docling)
------------------------------------------------------------------------------
Processo persistente que carrega UM motor pesado (PaddleOCR PP-Structure OU
Docling) uma única vez e atende requisições de extração via protocolo de linhas
JSON em stdin/stdout. Existe para contornar a incompatibilidade de runtime entre
`torch` (cu13, usado pelo Docling) e `paddlepaddle-gpu` (cu126): cada engine roda
no SEU venv, em SEU processo, opcionalmente fixado em UMA GPU via
`CUDA_VISIBLE_DEVICES` (definido pelo processo pai).

Protocolo (uma linha JSON por mensagem):
  stdout (1ª linha):  {"event":"ready","engine":..,"device":..,"load_s":..,"vram_gb":..}
  stdin  (requisição): {"id":N,"cmd":"extract","pdf":"...","pages":[..]|null}
                       {"cmd":"ping"} | {"cmd":"shutdown"}
  stdout (resposta):  {"id":N,"ok":bool,"chars":int,"text":str,"elapsed":float,
                       "vram_gb":float,"rss_gb":float,"error":str|null}

Toda a telemetria/diagnóstico vai para STDERR (não polui o canal de protocolo).

Uso (normalmente disparado pelo orquestrador, não à mão):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ./.venv-paddle/bin/python \
      -m dompi_scraper.engine_worker --engine paddle --device gpu --dpi 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import time
import traceback
from pathlib import Path

# Logging SEMPRE para stderr — stdout é reservado ao protocolo JSON.
log = logging.getLogger("engine_worker")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [worker:%(engine)s gpu=%(gpu)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    log.setLevel(level)
    log.handlers.clear()
    log.addHandler(handler)


class _CtxFilter(logging.Filter):
    """Injeta engine/gpu em todas as linhas de log para rastreabilidade."""

    def __init__(self, engine: str, gpu: str):
        super().__init__()
        self.engine = engine
        self.gpu = gpu

    def filter(self, record: logging.LogRecord) -> bool:
        record.engine = self.engine
        record.gpu = self.gpu
        return True


# ---------------------------------------------------------------------------
# MEDIÇÃO DE MEMÓRIA
# ---------------------------------------------------------------------------

def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _vram_reset(engine: str) -> None:
    try:
        if engine == "docling":
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        elif engine == "paddle":
            import paddle
            if paddle.is_compiled_with_cuda() and hasattr(paddle.device.cuda, "reset_max_memory_reserved"):
                paddle.device.cuda.reset_max_memory_reserved(0)
    except Exception as exc:  # noqa: BLE001
        log.debug("vram_reset falhou: %s", exc)


def _vram_gb(engine: str) -> float:
    try:
        if engine == "docling":
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_reserved() / 1e9
        elif engine == "paddle":
            import paddle
            if paddle.is_compiled_with_cuda():
                fn = getattr(paddle.device.cuda, "max_memory_reserved", None) or \
                     getattr(paddle.device.cuda, "memory_reserved", None)
                if fn:
                    return fn(0) / 1e9
    except Exception as exc:  # noqa: BLE001
        log.debug("vram_gb falhou: %s", exc)
    return 0.0


def _log_gpu_banner(engine: str) -> None:
    """Loga o que o worker enxerga de GPU (diagnóstico do conflito/pinning)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<não-definido>")
    log.info("CUDA_VISIBLE_DEVICES=%s", cvd)
    try:
        if engine == "docling":
            import torch
            log.info("torch=%s cuda_available=%s device_count=%s",
                     torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                log.info("GPU visível[0]: %s | VRAM total=%.1fGB", p.name, p.total_memory / 1e9)
        elif engine == "paddle":
            import paddle
            log.info("paddle=%s compiled_with_cuda=%s device_count=%s",
                     paddle.__version__, paddle.is_compiled_with_cuda(),
                     paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("Falha ao logar banner GPU: %s", exc)


# ---------------------------------------------------------------------------
# CARGA DA ENGINE
# ---------------------------------------------------------------------------

def _load_paddle(use_gpu: bool):
    from dompi_scraper.extrator_paddle import criar_engine_paddle
    log.info("Carregando PaddleOCR PP-Structure (use_gpu=%s)...", use_gpu)
    return criar_engine_paddle(use_gpu=use_gpu)


class _DoclingCache:
    """
    Mantém conversores Docling por modo de OCR. O custo de OCR do Docling vem do
    RapidOCR (onnxruntime, CPU); por isso o caminho rápido é `do_ocr=False`, que
    lê a CAMADA DE TEXTO nativa e usa a GPU só para layout/tabelas. O modo
    `do_ocr=True` é construído sob demanda (raro: páginas realmente escaneadas).
    """

    def __init__(self, use_gpu: bool):
        self.use_gpu = use_gpu
        self._cache: dict[bool, object] = {}

    def get(self, ocr: bool):
        if ocr not in self._cache:
            from dompi_scraper.extrator_docling import create_docling_session
            log.info("Construindo conversor Docling (do_ocr=%s, device=%s)...",
                     ocr, "gpu" if self.use_gpu else "cpu")
            self._cache[ocr] = create_docling_session(
                device="gpu" if self.use_gpu else "cpu",
                do_ocr=ocr, do_table_structure=True,
            )
        return self._cache[ocr]


def _load_docling(use_gpu: bool):
    mgr = _DoclingCache(use_gpu)
    mgr.get(False)  # pré-carrega o caminho rápido (texto nativo + layout GPU)
    return mgr


# ---------------------------------------------------------------------------
# EXTRAÇÃO POR REQUISIÇÃO
# ---------------------------------------------------------------------------

def _extract_paddle(engine_obj, pdf: str, pages, dpi: int, ocr: bool = True) -> str:
    """Extrai via PaddleOCR. Recorta mini-PDF se 'pages' for subconjunto."""
    import fitz
    import tempfile
    from dompi_scraper.extrator_paddle import extrair_pdf_paddle

    target = pdf
    tmp = None
    try:
        if pages:
            doc = fitz.open(pdf)
            total = doc.page_count
            if 0 < len(pages) < total:
                doc.select(sorted(set(pages)))
                fd, tmp = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                doc.save(tmp)
                log.debug("mini-PDF paddle: %d/%d págs -> %s", len(pages), total, tmp)
                target = tmp
            doc.close()
        res = extrair_pdf_paddle(engine_obj, target, dpi=dpi)
        return res.texto_completo
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def _extract_docling(engine_obj, pdf: str, pages, dpi: int, ocr: bool = False) -> str:
    """Extrai via Docling no modo de OCR pedido (False = caminho rápido nativo)."""
    from dompi_scraper.worker_docling import extrair_com_docling
    conv = engine_obj.get(ocr) if isinstance(engine_obj, _DoclingCache) else engine_obj
    return extrair_com_docling(conv, pdf, pages=pages or None)


_EXTRACTORS = {"paddle": _extract_paddle, "docling": _extract_docling}
_LOADERS = {"paddle": _load_paddle, "docling": _load_docling}


# ---------------------------------------------------------------------------
# LOOP PRINCIPAL DO WORKER
# ---------------------------------------------------------------------------

def _send(obj: dict) -> None:
    """Escreve uma resposta JSON (uma linha) no stdout e dá flush."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_worker(engine: str, device: str, dpi: int) -> int:
    use_gpu = device == "gpu"

    _log_gpu_banner(engine)

    # --- carga única dos modelos ---
    t0 = time.time()
    try:
        _vram_reset(engine)
        engine_obj = _LOADERS[engine](use_gpu)
    except Exception as exc:  # noqa: BLE001
        log.error("FALHA ao carregar engine %s: %s\n%s", engine, exc, traceback.format_exc())
        _send({"event": "ready", "engine": engine, "ok": False, "error": str(exc)})
        return 1

    load_s = time.time() - t0
    vram = _vram_gb(engine)
    log.info("Engine %s pronta em %.1fs | VRAM=%.2fGB | RSS=%.2fGB",
             engine, load_s, vram, _rss_gb())
    _send({"event": "ready", "engine": engine, "ok": True, "device": device,
           "load_s": round(load_s, 2), "vram_gb": round(vram, 2)})

    # --- loop de requisições ---
    n_req = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("Requisição inválida ignorada: %s (%s)", line[:120], exc)
            continue

        cmd = req.get("cmd", "extract")
        if cmd == "shutdown":
            log.info("Shutdown solicitado após %d requisições.", n_req)
            _send({"event": "bye", "served": n_req})
            return 0
        if cmd == "ping":
            _send({"event": "pong"})
            continue

        rid = req.get("id")
        pdf = req.get("pdf", "")
        pages = req.get("pages")
        ocr = req.get("ocr", False)
        n_req += 1

        log.info("[req %s] extract pdf=%s pages=%s ocr=%s",
                 rid, os.path.basename(pdf), f"{len(pages)}p" if pages else "todas", ocr)

        if not pdf or not os.path.exists(pdf):
            log.error("[req %s] PDF inexistente: %s", rid, pdf)
            _send({"id": rid, "ok": False, "error": f"pdf inexistente: {pdf}",
                   "chars": 0, "text": ""})
            continue

        _vram_reset(engine)
        t1 = time.time()
        try:
            text = _EXTRACTORS[engine](engine_obj, pdf, pages, dpi, ocr) or ""
            elapsed = time.time() - t1
            vram = _vram_gb(engine)
            rss = _rss_gb()
            log.info("[req %s] OK %d chars em %.1fs | VRAM=%.2fGB | RSS=%.2fGB",
                     rid, len(text), elapsed, vram, rss)
            _send({"id": rid, "ok": True, "chars": len(text), "text": text,
                   "elapsed": round(elapsed, 2), "vram_gb": round(vram, 2),
                   "rss_gb": round(rss, 2), "error": None})
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t1
            tb = traceback.format_exc()
            log.error("[req %s] FALHA em %.1fs: %s\n%s", rid, elapsed, exc, tb)
            _send({"id": rid, "ok": False, "chars": 0, "text": "",
                   "elapsed": round(elapsed, 2), "error": f"{type(exc).__name__}: {exc}"})

    log.info("stdin encerrado — worker saindo após %d requisições.", n_req)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Worker de extração isolado (paddle|docling).")
    ap.add_argument("--engine", required=True, choices=["paddle", "docling"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _configure_logging(args.verbose)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for h in log.handlers:
        h.addFilter(_CtxFilter(args.engine, cvd))

    log.info("Worker iniciando: engine=%s device=%s dpi=%d pid=%d",
             args.engine, args.device, args.dpi, os.getpid())
    rc = run_worker(args.engine, args.device, args.dpi)
    sys.exit(rc)


if __name__ == "__main__":
    main()
