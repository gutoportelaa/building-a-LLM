#!/usr/bin/env python3
"""
worker_client.py — Cliente do orquestrador para os engine_workers isolados
--------------------------------------------------------------------------
Gerencia o ciclo de vida de um `engine_worker.py` rodando em OUTRO venv e
(opcionalmente) fixado em UMA GPU via CUDA_VISIBLE_DEVICES. Faz o handshake de
prontidão, envia requisições de extração e recebe o texto, drenando o stderr do
worker para o log do orquestrador (diagnóstico unificado).

Motivação: `torch` (cu13/Docling) e `paddlepaddle-gpu` (cu126) não coexistem no
mesmo processo/venv. Cada engine vive num WorkerClient próprio.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("orquestrador")


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    """Subprocesso persistente de extração para uma engine (paddle|docling)."""

    def __init__(
        self,
        engine: str,
        python_exe: str,
        repo_root: Path,
        gpu_id: str | None,
        dpi: int = 200,
        verbose: bool = True,
        ready_timeout: float = 600.0,
        req_timeout: float = 1800.0,
    ):
        self.engine = engine
        self.python_exe = python_exe
        self.repo_root = Path(repo_root)
        self.gpu_id = gpu_id            # "0"/"1" para GPU, None/"" para CPU
        self.dpi = dpi
        self.verbose = verbose
        self.ready_timeout = ready_timeout
        self.req_timeout = req_timeout

        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._rid = 0
        self.ready_info: dict | None = None

    # ------------------------------------------------------------------ spawn
    def start(self) -> dict:
        device = "gpu" if self.gpu_id not in (None, "") else "cpu"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        # Pinning de GPU: isola o worker numa única placa (ou nenhuma)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_id if device == "gpu" else ""
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [
            self.python_exe, "-m", "dompi_scraper.engine_worker",
            "--engine", self.engine, "--device", device, "--dpi", str(self.dpi),
        ]
        if self.verbose:
            cmd.append("--verbose")

        log.info("[%s] spawn: CUDA_VISIBLE_DEVICES=%r device=%s exe=%s",
                 self.engine, env["CUDA_VISIBLE_DEVICES"], device, self.python_exe)
        log.debug("[%s] cmd: %s", self.engine, " ".join(cmd))

        self.proc = subprocess.Popen(
            cmd, env=env, cwd=str(self.repo_root),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

        # handshake de prontidão (carga de modelos pode levar minutos)
        info = self._await(lambda m: m.get("event") == "ready", self.ready_timeout,
                           what="ready")
        if not info.get("ok"):
            raise WorkerError(f"[{self.engine}] worker não ficou pronto: {info.get('error')}")
        self.ready_info = info
        log.info("[%s] PRONTO em %.1fs | VRAM_carga=%.2fGB",
                 self.engine, info.get("load_s", 0.0), info.get("vram_gb", 0.0))
        return info

    # ------------------------------------------------------------------ pumps
    def _pump_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._q.put(json.loads(line))
            except json.JSONDecodeError:
                log.debug("[%s] stdout não-JSON: %s", self.engine, line[:160])

    def _pump_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            line = line.rstrip()
            if line:
                # diagnóstico do worker rebaixado a DEBUG no log do orquestrador
                log.debug("[%s::worker] %s", self.engine, line)

    # ------------------------------------------------------------------ io
    def _await(self, pred, timeout: float, what: str) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise WorkerError(f"[{self.engine}] worker morreu (rc={self.proc.returncode}) aguardando {what}")
            try:
                msg = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if pred(msg):
                return msg
            # mensagens fora de ordem (eventos) são apenas logadas
            log.debug("[%s] msg descartada aguardando %s: %s", self.engine, what, msg)
        raise WorkerError(f"[{self.engine}] timeout ({timeout}s) aguardando {what}")

    def extract(self, pdf: str, pages: list[int] | None = None,
                ocr: bool = False) -> dict:
        """Envia uma requisição de extração e retorna a resposta do worker.

        `ocr` só afeta o Docling: False = caminho rápido (texto nativo + layout
        GPU); True = liga o RapidOCR (páginas escaneadas). PaddleOCR ignora.
        """
        if not self.proc or self.proc.poll() is not None:
            raise WorkerError(f"[{self.engine}] worker não está vivo")
        self._rid += 1
        rid = self._rid
        req = {"id": rid, "cmd": "extract", "pdf": pdf, "pages": pages, "ocr": ocr}
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        log.debug("[%s] -> req %d pdf=%s pages=%s", self.engine, rid,
                  os.path.basename(pdf), f"{len(pages)}p" if pages else "todas")
        resp = self._await(lambda m: m.get("id") == rid, self.req_timeout, what=f"resp#{rid}")
        return resp

    # ------------------------------------------------------------------ stop
    def stop(self) -> None:
        if not self.proc:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if self.proc.poll() is None:
                log.warning("[%s] forçando término do worker", self.engine)
                self.proc.kill()
            log.info("[%s] worker encerrado (rc=%s)", self.engine, self.proc.returncode)
