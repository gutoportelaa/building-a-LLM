#!/usr/bin/env python3
"""
build_corpus.py — LIMPO → CORPUS/corpus_llm (pronto para treino de LLM).

A camada corpus materializa o produto de consumo priorizado (corpus LLM). Lê a limpo
inteira (já limpa e deduplicada por id_limpo) e produz:

  • Parquet particionado por `ano`     → datalake/corpus/corpus_llm/ano=<AAAA>/*.parquet
  • shards .jsonl.zst (formato treino) → datalake/corpus/corpus_llm/shards/part-*.jsonl.zst

Filtros de qualidade (configuráveis):
  --min-chars N    descarta textos com menos de N caracteres limpos (padrão 1)
  --drop-review    exclui documentos com needs_human_review=true (padrão: mantém)

Colunas do corpus: id, territorio, municipio, tipo_ato, ano, n_tokens, texto.

Uso:
    python -m dompi_scraper.datalake.build_corpus
    python -m dompi_scraper.datalake.build_corpus --drop-review --min-chars 200 --shard-size 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import polars as pl
import zstandard as zstd

from . import ensure_zones, zone_dir
from .io import read_zone, write_partitioned_parquet

log = logging.getLogger("build_corpus")

_CORPUS_COLUMNS = ["id", "territorio", "municipio", "tipo_ato", "ano", "n_tokens", "texto"]


def _write_jsonl_zst(df: pl.DataFrame, dest_dir: Path, shard_size: int, level: int = 10) -> int:
    """Escreve o corpus em shards .jsonl.zst de até `shard_size` linhas. Retorna nº de shards."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=level)
    n_shards = 0
    for i in range(0, df.height, shard_size):
        shard = df.slice(i, shard_size)
        path = dest_dir / f"part-{n_shards:05d}.jsonl.zst"
        with open(path, "wb") as fh, cctx.stream_writer(fh) as z:
            for row in shard.iter_rows(named=True):
                z.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        n_shards += 1
    return n_shards


def build_corpus(
    root=None, min_chars: int = 1, drop_review: bool = False, shard_size: int = 5000
) -> dict:
    limpo = read_zone(zone_dir("limpo", root))
    if limpo.is_empty():
        raise FileNotFoundError("Limpo vazio. Rode build_limpo antes.")
    n_limpo = limpo.height

    corpus = limpo.filter(pl.col("n_chars_limpo") >= min_chars)
    if drop_review:
        corpus = corpus.filter(~pl.col("needs_human_review"))
    # Dedup global defensivo (limpo já é distinto por id_limpo) + projeção final.
    corpus = corpus.unique(subset=["id_limpo"], keep="first").select(
        pl.col("id_limpo").alias("id"),
        "territorio", "municipio", "tipo_ato", "ano", "n_tokens",
        pl.col("texto_limpo").alias("texto"),
    )
    n_corpus = corpus.height

    ensure_zones(root)
    corpus_dir = zone_dir("corpus", root) / "corpus_llm"
    write_partitioned_parquet(corpus, corpus_dir, ["ano"])
    n_shards = _write_jsonl_zst(corpus, corpus_dir / "shards", shard_size)

    total_tokens = int(corpus["n_tokens"].sum())
    log.info("Corpus corpus_llm: %d/%d docs, ~%d tokens, %d shards", n_corpus, n_limpo, total_tokens, n_shards)
    return {
        "limpo": n_limpo,
        "corpus": n_corpus,
        "descartados": n_limpo - n_corpus,
        "tokens_estimados": total_tokens,
        "shards": n_shards,
        "dir": str(corpus_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Materializa o corpus LLM (corpus) a partir da limpo.")
    ap.add_argument("--root", default=None, help="Raiz do data lake (padrão: ./datalake).")
    ap.add_argument("--min-chars", type=int, default=1, help="Descarta textos abaixo deste tamanho.")
    ap.add_argument("--drop-review", action="store_true", help="Exclui needs_human_review=true.")
    ap.add_argument("--shard-size", type=int, default=5000, help="Docs por shard .jsonl.zst.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )
    res = build_corpus(args.root, args.min_chars, args.drop_review, args.shard_size)
    print(
        f"\n  Limpo:            {res['limpo']}\n"
        f"  Corpus (corpus_llm): {res['corpus']}\n"
        f"  Descartados:       {res['descartados']}\n"
        f"  Tokens estimados:  ~{res['tokens_estimados']:,}\n"
        f"  Shards .jsonl.zst: {res['shards']}\n"
        f"  Saída:             {res['dir']}\n"
    )


if __name__ == "__main__":
    sys.exit(main())
