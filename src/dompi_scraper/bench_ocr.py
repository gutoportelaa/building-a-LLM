#!/usr/bin/env python3
"""
bench_ocr.py — Benchmark comparativo de backends de extração DOM-PI
-------------------------------------------------------------------
Compara Docling e PaddleOCR (PP-Structure) em GPU e CPU sobre um conjunto de
PDFs, medindo para cada execução:

  - tempo de carga da sessão (modelos) e tempo por PDF
  - pico de RAM do processo (RSS, via amostragem psutil + ru_maxrss)
  - pico de VRAM (contadores internos torch/paddle — NVML/nvidia-smi não são
    necessários, o que é essencial nesta máquina onde o NVML está quebrado)
  - tamanho do texto extraído e score médio de qualidade OCR (heurística do
    projeto, reaproveitada de processar_pdfs.compute_ocr_quality_score)
  - falhas / OOM (cada combinação engine×device roda em SUBPROCESSO isolado,
    então um estouro de memória não derruba o restante do benchmark)

Orquestrador (roda todas as combinações pedidas):
    uv run python -m dompi_scraper.bench_ocr \
        --pdfs "teste_extractor/*.pdf" \
        --engines docling-cpu docling-gpu paddle-cpu paddle-gpu \
        --do-ocr --out dados_benchmark/resultado.json

Worker (uso interno — uma combinação por processo):
    uv run python -m dompi_scraper.bench_ocr --worker \
        --engine docling --device gpu --pdfs a.pdf b.pdf --do-ocr
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from dompi_scraper.processar_pdfs import compute_ocr_quality_score


# ===========================================================================
# AMOSTRAGEM DE MEMÓRIA
# ===========================================================================

class RSSPeakSampler:
    """Amostra o RSS do processo numa thread, registrando o pico observado."""

    def __init__(self, interval: float = 0.1):
        self._proc = psutil.Process()
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                # inclui filhos (paddle/docling podem usar subprocessos/threads)
                for child in self._proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except psutil.Error:
                rss = 0
            self._peak = max(self._peak, rss)
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_gb(self) -> float:
        return self._peak / 1e9


def _gpu_peak_reset(framework: str):
    """Zera os contadores de pico de VRAM antes de uma medição."""
    try:
        if framework == "torch":
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
        elif framework == "paddle":
            import paddle
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                # reset_max_memory_* nem sempre existe; ignorado se ausente
                if hasattr(paddle.device.cuda, "reset_max_memory_allocated"):
                    paddle.device.cuda.reset_max_memory_allocated(0)
                    paddle.device.cuda.reset_max_memory_reserved(0)
    except Exception:  # noqa: BLE001
        pass


def _gpu_peak_gb(framework: str) -> float:
    """Lê o pico de VRAM reservada (GB) pelos contadores internos do framework."""
    try:
        if framework == "torch":
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_reserved() / 1e9
        elif framework == "paddle":
            import paddle
            if paddle.is_compiled_with_cuda():
                if hasattr(paddle.device.cuda, "max_memory_reserved"):
                    return paddle.device.cuda.max_memory_reserved(0) / 1e9
                if hasattr(paddle.device.cuda, "memory_reserved"):
                    return paddle.device.cuda.memory_reserved(0) / 1e9
    except Exception:  # noqa: BLE001
        pass
    return 0.0


# ===========================================================================
# WORKER — processa um conjunto de PDFs com uma engine/device
# ===========================================================================

def run_worker(engine: str, device: str, pdfs: list[str], do_ocr: bool,
               dpi: int, page_batch: int | None) -> dict:
    """Executa a extração e devolve métricas. Pensado para rodar em subprocesso."""
    framework = "torch" if engine == "docling" else "paddle"
    use_gpu = device == "gpu"

    result: dict = {
        "engine": engine, "device": device, "do_ocr": do_ocr, "dpi": dpi,
        "page_batch": page_batch, "ok": False, "error": None,
        "load_s": 0.0, "pdfs": [],
    }

    _gpu_peak_reset(framework)

    # --- carrega a sessão (modelos) uma única vez ---
    try:
        t0 = time.time()
        if engine == "docling":
            from dompi_scraper.extrator_docling import create_docling_session
            session = create_docling_session(
                device=device, do_ocr=do_ocr, page_batch=page_batch,
            )
            extract = _make_docling_extract(session)
        elif engine == "paddle":
            from dompi_scraper.extrator_paddle import criar_engine_paddle, extrair_pdf_paddle
            session = criar_engine_paddle(use_gpu=use_gpu)
            extract = lambda p: extrair_pdf_paddle(session, p, dpi=dpi).texto_completo  # noqa: E731
        else:
            raise ValueError(f"engine desconhecida: {engine}")
        result["load_s"] = round(time.time() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"load: {type(exc).__name__}: {exc}"
        return result

    # --- processa cada PDF medindo tempo e memória ---
    with RSSPeakSampler() as sampler:
        for pdf in pdfs:
            entry = {"pdf": os.path.basename(pdf), "pages": _page_count(pdf)}
            try:
                t1 = time.time()
                text = extract(pdf) or ""
                entry["time_s"] = round(time.time() - t1, 2)
                entry["chars"] = len(text)
                entry["ocr_score"] = round(compute_ocr_quality_score(text[:5000]), 3)
                entry["ok"] = True
            except Exception as exc:  # noqa: BLE001
                entry["ok"] = False
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            result["pdfs"].append(entry)

        result["peak_rss_gb"] = round(sampler.peak_gb, 2)

    result["peak_vram_gb"] = round(_gpu_peak_gb(framework), 2)
    result["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    result["ok"] = any(p.get("ok") for p in result["pdfs"])
    return result


def _make_docling_extract(session):
    from dompi_scraper.extrator_docling import extract_with_docling
    return lambda p: extract_with_docling(session, p)


def _page_count(pdf: str) -> int:
    try:
        import fitz
        d = fitz.open(pdf)
        n = d.page_count
        d.close()
        return n
    except Exception:  # noqa: BLE001
        return -1


# ===========================================================================
# ORQUESTRADOR — dispara um subprocesso por combinação engine×device
# ===========================================================================

def run_orchestrator(args) -> None:
    pdfs = _resolve_pdfs(args.pdfs, args.limite)
    if not pdfs:
        print("Nenhum PDF encontrado.", file=sys.stderr)
        sys.exit(1)

    print(f"== Benchmark OCR DOM-PI ==  PDFs={len(pdfs)}  engines={args.engines}")
    print(f"   do_ocr={args.do_ocr} dpi={args.dpi} page_batch={args.page_batch}\n")

    repo_root = Path(__file__).resolve().parents[2]
    py_docling = args.python_docling or str(repo_root / ".venv" / "bin" / "python")
    py_paddle = args.python_paddle or str(repo_root / ".venv-paddle" / "bin" / "python")

    all_results = []
    for combo in args.engines:
        engine, device = combo.split("-")
        print(f"[{combo}] subprocesso iniciando ...", flush=True)
        interp = py_paddle if engine == "paddle" else py_docling
        cmd = [
            interp, "-m", "dompi_scraper.bench_ocr", "--worker",
            "--engine", engine, "--device", device,
            "--dpi", str(args.dpi),
            "--pdfs", *pdfs,
        ]
        if args.do_ocr:
            cmd.append("--do-ocr")
        if args.page_batch is not None:
            cmd += ["--page-batch", str(args.page_batch)]

        # PYTHONPATH=src garante que dompi_scraper seja importável mesmo em
        # ambientes onde o pacote não foi instalado em modo editável (.venv-paddle)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout, env=env)
        wall = time.time() - t0

        res = _parse_worker_output(proc, combo, wall)
        all_results.append(res)
        _print_combo_summary(res, wall)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nResultados salvos em {args.out}")

    _print_final_table(all_results)


def _parse_worker_output(proc, combo: str, wall: float) -> dict:
    marker = "===BENCH_JSON==="
    if marker in proc.stdout:
        try:
            payload = proc.stdout.split(marker, 1)[1].strip()
            res = json.loads(payload)
            res["wall_s"] = round(wall, 1)
            return res
        except json.JSONDecodeError:
            pass
    engine, device = combo.split("-")
    return {
        "engine": engine, "device": device, "ok": False, "wall_s": round(wall, 1),
        "error": f"sem JSON (rc={proc.returncode}). stderr: {proc.stderr[-400:]}",
        "pdfs": [],
    }


def _print_combo_summary(res: dict, wall: float) -> None:
    if res.get("error"):
        print(f"   -> FALHA: {res['error']}\n")
        return
    oks = [p for p in res["pdfs"] if p.get("ok")]
    fails = [p for p in res["pdfs"] if not p.get("ok")]
    avg_t = sum(p["time_s"] for p in oks) / len(oks) if oks else 0
    print(f"   -> ok={len(oks)} fail={len(fails)} load={res.get('load_s')}s "
          f"avg/pdf={avg_t:.1f}s RAM_pico={res.get('peak_rss_gb')}GB "
          f"VRAM_pico={res.get('peak_vram_gb')}GB wall={wall:.0f}s")
    for f in fails:
        print(f"      ! {f['pdf']}: {f.get('error')}")
    print()


def _print_final_table(results: list[dict]) -> None:
    print("\n" + "=" * 88)
    print(f"{'engine-device':<16}{'ok/fail':>9}{'load_s':>8}{'avg_s/pdf':>11}"
          f"{'RAM_GB':>9}{'VRAM_GB':>9}{'avg_chars':>11}{'avg_score':>10}")
    print("-" * 88)
    for r in results:
        combo = f"{r['engine']}-{r['device']}"
        if r.get("error") and not r["pdfs"]:
            print(f"{combo:<16}{'ERRO':>9}  {r['error'][:60]}")
            continue
        oks = [p for p in r["pdfs"] if p.get("ok")]
        fails = [p for p in r["pdfs"] if not p.get("ok")]
        avg_t = sum(p["time_s"] for p in oks) / len(oks) if oks else 0
        avg_c = sum(p["chars"] for p in oks) / len(oks) if oks else 0
        avg_s = sum(p["ocr_score"] for p in oks) / len(oks) if oks else 0
        print(f"{combo:<16}{f'{len(oks)}/{len(fails)}':>9}{r.get('load_s',0):>8.1f}"
              f"{avg_t:>11.1f}{r.get('peak_rss_gb',0):>9.1f}{r.get('peak_vram_gb',0):>9.1f}"
              f"{avg_c:>11.0f}{avg_s:>10.3f}")
    print("=" * 88)


def _resolve_pdfs(patterns: list[str], limite: int | None) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        if os.path.isfile(pat):
            out.append(pat)
        else:
            out.extend(sorted(glob.glob(pat, recursive=True)))
    # dedup preservando ordem
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:limite] if limite else uniq


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="Modo worker (uso interno).")
    ap.add_argument("--engine", choices=["docling", "paddle"], help="(worker) engine.")
    ap.add_argument("--device", choices=["cpu", "gpu"], help="(worker) dispositivo.")
    ap.add_argument("--engines", nargs="+",
                    default=["docling-cpu", "docling-gpu", "paddle-cpu", "paddle-gpu"],
                    help="(orquestrador) combinações engine-device.")
    ap.add_argument("--pdfs", nargs="+", required=True, help="Globs ou caminhos de PDFs.")
    ap.add_argument("--limite", type=int, default=None, help="Máx. de PDFs.")
    ap.add_argument("--do-ocr", action="store_true", help="Aciona OCR (Docling).")
    ap.add_argument("--dpi", type=int, default=200, help="DPI de rasterização (Paddle).")
    ap.add_argument("--page-batch", type=int, default=None, help="Páginas por lote (Docling, anti-OOM).")
    ap.add_argument("--timeout", type=int, default=3600, help="Timeout por subprocesso (s).")
    ap.add_argument("--python-docling", default=None, help="Interpretador p/ docling (default: .venv).")
    ap.add_argument("--python-paddle", default=None, help="Interpretador p/ paddle (default: .venv-paddle).")
    ap.add_argument("--out", default=None, help="Arquivo JSON de saída (orquestrador).")
    args = ap.parse_args()

    if args.worker:
        res = run_worker(args.engine, args.device, args.pdfs, args.do_ocr,
                         args.dpi, args.page_batch)
        print("===BENCH_JSON===")
        print(json.dumps(res, ensure_ascii=False))
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
