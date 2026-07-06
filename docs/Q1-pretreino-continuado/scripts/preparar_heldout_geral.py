#!/usr/bin/env python3
"""
preparar_heldout_geral.py — Held-out de DOMÍNIO GERAL (camada "retenção de capacidade").

Para evidenciar que o DAPT não causou esquecimento catastrófico, precisamos de
um corpus FORA do domínio DOM-PI. Medimos PPL/CE do baseline e do modelo pós-DAPT
neste held-out: se a PPL geral subir pouco (ex.: <5-10%) enquanto a PPL de domínio
cai bastante, o DAPT foi bem-sucedido (ganho de domínio > custo de retenção) —
exatamente o trade-off que o Juru (arXiv:2403.18140) reporta para PT-BR.

Fonte padrão: Wikipedia em português (texto limpo, claramente fora de diários
oficiais). Em streaming para não baixar o dump inteiro.

Uso:
    python scripts/preparar_heldout_geral.py --n 1000 \
        --output data/held_out_geral.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# datasets candidatos em ordem de preferência (config, split)
CANDIDATES = [
    ("wikimedia/wikipedia", "20231101.pt", "train", "text"),
    ("graelo/wikipedia", "20230901.pt", "train", "text"),
]


def stream_texts(n: int, min_chars: int):
    from datasets import load_dataset

    last_err = None
    for repo, config, split, field in CANDIDATES:
        try:
            log.info("Tentando %s [%s]...", repo, config)
            ds = load_dataset(repo, config, split=split, streaming=True)
            out = []
            for ex in ds:
                txt = (ex.get(field) or "").strip()
                if len(txt) < min_chars:
                    continue
                # limita tamanho por doc p/ casar com docs de diário (~2-4k chars)
                out.append(txt[:4000])
                if len(out) >= n:
                    break
            if out:
                log.info("OK: %d docs de %s", len(out), repo)
                return out
        except Exception as e:  # noqa: BLE001
            log.warning("Falhou %s: %s", repo, e)
            last_err = e
    raise RuntimeError(f"Nenhum dataset geral disponível. Último erro: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out de domínio geral (retenção)")
    parser.add_argument("--n", type=int, default=1000, help="Número de documentos")
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--output", default="data/held_out_geral.jsonl")
    args = parser.parse_args()

    texts = stream_texts(args.n, args.min_chars)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in texts:
            f.write(json.dumps({"texto": t}, ensure_ascii=False) + "\n")
    log.info("Held-out geral salvo: %d docs em %s", len(texts), out)
    log.info("Avalie com: python scripts/avaliar_modelo.py --model <m> "
             "--held-out %s --benchmark dompi_qa.jsonl --output resultados/retencao_<m>.json", out)
    # O iterador streaming do HF deixa threads de fundo que podem dar core dump
    # (PyGILState_Release) no teardown do interpretador. Tudo já foi gravado em
    # disco; encerramos de forma limpa pulando os finalizadores problemáticos.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
