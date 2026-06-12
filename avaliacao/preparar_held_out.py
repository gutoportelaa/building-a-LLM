#!/usr/bin/env python3
"""
preparar_held_out.py — Separa held-out e treino do corpus curated DOM-PI.

Reserva ~5% dos documentos como held-out fixo (reproduzível via seed),
garantindo que os IDs do held-out não apareçam no treino.

Fontes suportadas:
  --corpus-dir   : Parquet local (padrão, uso local)
  --from-hub     : Carrega config 'curated' direto do HF Hub (uso no cluster)

Saída:
    data/held_out.jsonl      (~5% dos docs Tier A+B)
    data/train_corpus.jsonl  (restante, Tier A+B)

Uso local (Parquet):
    python avaliacao/preparar_held_out.py

Uso no cluster (HF Hub, sem transferência de arquivo):
    python avaliacao/preparar_held_out.py --from-hub
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_REPO = "gutoportelaa/dom-pi-corpus-2025"


def load_from_parquet(corpus_dir: str) -> pl.DataFrame:
    parquets = list(Path(corpus_dir).glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"Nenhum .parquet em {corpus_dir}")
    log.info("Carregando %d arquivos Parquet locais...", len(parquets))
    return pl.concat([pl.read_parquet(p) for p in sorted(parquets)])


def load_from_hub(repo: str) -> pl.DataFrame:
    from datasets import load_dataset
    log.info("Baixando config 'curated' de %s ...", repo)
    ds = load_dataset(repo, "curated", split="train", trust_remote_code=True)
    log.info("Dataset carregado: %d exemplos", len(ds))
    return pl.from_arrow(ds.data.table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="hf_corpus_dompi/curated",
                        help="Diretório com Parquets locais (ignorado se --from-hub)")
    parser.add_argument("--from-hub", action="store_true",
                        help="Carrega direto do HuggingFace Hub (cluster, sem arquivos locais)")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--held-out-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiers", default="A,B")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    df = load_from_hub(args.hf_repo) if args.from_hub else load_from_parquet(args.corpus_dir)
    log.info("Total de docs: %d", len(df))

    tiers = [t.strip() for t in args.tiers.split(",")]
    df = df.filter(pl.col("quality_tier").is_in(tiers))
    log.info("Após filtro tier (%s): %d docs", args.tiers, len(df))

    if args.max_tokens:
        df = df.filter(pl.col("n_tokens") <= args.max_tokens)
        log.info("Após filtro n_tokens <= %d: %d docs", args.max_tokens, len(df))

    df = df.sample(fraction=1.0, shuffle=True, seed=args.seed)
    n_held = int(len(df) * args.held_out_frac)
    held_out = df[:n_held]
    train = df[n_held:]
    log.info("Held-out: %d docs | Treino: %d docs", len(held_out), len(train))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ho_path = out_dir / "held_out.jsonl"
    tr_path = out_dir / "train_corpus.jsonl"

    log.info("Escrevendo %s...", ho_path)
    with ho_path.open("w", encoding="utf-8") as f:
        for row in held_out.iter_rows(named=True):
            f.write(json.dumps({"id": row["id"], "texto": row["texto"]}, ensure_ascii=False) + "\n")

    log.info("Escrevendo %s...", tr_path)
    with tr_path.open("w", encoding="utf-8") as f:
        for row in train.iter_rows(named=True):
            f.write(json.dumps({"id": row["id"], "texto": row["texto"]}, ensure_ascii=False) + "\n")

    ids_path = out_dir / "held_out_ids.txt"
    ids_path.write_text("\n".join(held_out["id"].to_list()))

    log.info("Pronto. held_out=%d → %s | train=%d → %s", len(held_out), ho_path, len(train), tr_path)


if __name__ == "__main__":
    main()
