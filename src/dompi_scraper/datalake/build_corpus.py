#!/usr/bin/env python3
"""
build_corpus.py — LIMPO → CORPUS (pronto para treino de LLM), com D-1 e D-2.

A camada corpus materializa o produto de consumo. Lê a limpo inteira (já limpa e
deduplicada por id_limpo) e aplica:

  • D-2 (fatiar_megadocs): documentos > `--limite-split` tokens (padrão 8192) são
    fatiados em ATOS atômicos quando há títulos de ato (PORTARIA/DECRETO/LEI/… Nº);
    documentos únicos (orçamento, planilha, uma lei) ficam intactos. Acrescenta a
    coluna `tamanho_classe` (normal/longo/mega).
  • D-1 (dedup_aproximada): MinHash+LSH marca quase-duplicatas; elege 1 canônico por
    cluster. Catálogo gravado em _catalog/near_dup.parquet.

Produz DOIS produtos:
  corpus/corpus_llm/  → split 'train': SÓ canônicos (near-dups removidos). Parquet por
                        ano + shards .jsonl.zst. É o dataset de treino.
  corpus/corpus_raw/  → split 'raw': TODOS os docs (pós-fatiamento) com cluster_id/
                        is_near_dup/is_canonical para auditoria e dedup própria.

Colunas (train): id, territorio, municipio, tipo_ato, ano, data_publicacao, n_tokens,
tamanho_classe, texto.

Uso:
    python -m dompi_scraper.datalake.build_corpus
    python -m dompi_scraper.datalake.build_corpus --no-near-dup            # só D-2
    python -m dompi_scraper.datalake.build_corpus --threshold 0.80 --limite-split 8192
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
from .fatiar_megadocs import expandir_megadocs, LIMITE_LONGO
from .dedup_aproximada import marcar_near_dups

log = logging.getLogger("build_corpus")

# Ordem das colunas do produto de treino (mantém paridade com o publicado no HF + novidades).
_TRAIN_COLUMNS = ["id", "territorio", "municipio", "tipo_ato", "ano",
                  "data_publicacao", "n_tokens", "tamanho_classe", "texto"]
_RAW_EXTRA = ["cluster_id", "is_near_dup", "is_canonical"]


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
    root=None, min_chars: int = 1, drop_review: bool = False, shard_size: int = 5000,
    limite_split: int = LIMITE_LONGO, near_dup: bool = True,
    threshold: float = 0.85, num_perm: int = 128,
) -> dict:
    limpo = read_zone(zone_dir("limpo", root))
    if limpo.is_empty():
        raise FileNotFoundError("Limpo vazio. Rode build_limpo antes.")
    n_limpo = limpo.height

    base = limpo.filter(pl.col("n_chars_limpo") >= min_chars)
    if drop_review:
        base = base.filter(~pl.col("needs_human_review"))
    keep_reasons = ["review_reasons"] if "review_reasons" in base.columns else []
    base = base.unique(subset=["id_limpo"], keep="first").select(
        pl.col("id_limpo").alias("id"),
        "territorio", "municipio", "tipo_ato", "ano",
        pl.col("data_publicacao").fill_null("").alias("data_publicacao"),
        "n_tokens",
        pl.col("texto_limpo").alias("texto"),
        *keep_reasons,
    )
    n_base = base.height

    # ── D-2: fatiar documentos longos em atos atômicos ───────────────────────
    expandido, st_split = expandir_megadocs(base, limite_split=limite_split)
    # fatias minúsculas e duplicatas EXATAS geradas pelo fatiamento (boilerplate repetido)
    expandido = expandido.filter(pl.col("texto").str.len_chars() >= min_chars)
    expandido = expandido.unique(subset=["id"], keep="first")
    n_pos_split = expandido.height
    log.info("D-2 fatiamento: %d docs → %d (fatiados=%d, fatias=%d, intactos=%d)",
             n_base, n_pos_split, st_split["docs_fatiados"], st_split["fatias_geradas"],
             st_split["docs_intactos"])

    # ── D-1: marcar quase-duplicatas (MinHash+LSH) ───────────────────────────
    if near_dup:
        marc = marcar_near_dups(expandido, threshold=threshold, num_perm=num_perm,
                                id_col="id", text_col="texto")
    else:
        marc = expandido.with_columns(
            cluster_id=pl.int_range(0, pl.len(), dtype=pl.Int64),
            is_near_dup=pl.lit(False),
            is_canonical=pl.lit(True),
        )
    n_redundantes = int((~marc["is_canonical"]).sum())

    ensure_zones(root)
    # catálogo de near-dup (auditoria/reversível) em datalake/_catalog/
    cat_dir = zone_dir("corpus", root).parent / "_catalog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    marc.select("id", "cluster_id", "is_near_dup", "is_canonical").write_parquet(
        cat_dir / "near_dup.parquet")

    # ── RAW: todos os docs pós-fatiamento (com flags) ────────────────────────
    raw = marc.select(*_TRAIN_COLUMNS, *_RAW_EXTRA)
    raw_dir = zone_dir("corpus", root) / "corpus_raw"
    write_partitioned_parquet(raw.drop("is_canonical"), raw_dir, ["ano"])

    # ── TRAIN: só canônicos ──────────────────────────────────────────────────
    train = marc.filter(pl.col("is_canonical")).select(_TRAIN_COLUMNS)
    n_train = train.height
    corpus_dir = zone_dir("corpus", root) / "corpus_llm"
    write_partitioned_parquet(train, corpus_dir, ["ano"])
    n_shards = _write_jsonl_zst(train, corpus_dir / "shards", shard_size)

    total_tokens = int(train["n_tokens"].sum())
    dist = (marc.group_by("tamanho_classe").agg(pl.len().alias("k"))
            .sort("k", descending=True).to_dicts())
    log.info("Corpus train: %d docs, ~%d tokens, %d shards", n_train, total_tokens, n_shards)
    return {
        "limpo": n_limpo,
        "base": n_base,
        "pos_split": n_pos_split,
        "fatiados": st_split["docs_fatiados"],
        "fatias": st_split["fatias_geradas"],
        "near_dup_removidos": n_redundantes,
        "train": n_train,
        "raw": raw.height,
        "tokens_estimados": total_tokens,
        "shards": n_shards,
        "tamanho_classe": {d["tamanho_classe"]: d["k"] for d in dist},
        "dir": str(corpus_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Materializa o corpus LLM (train+raw) com D-1/D-2.")
    ap.add_argument("--root", default=None, help="Raiz do data lake (padrão: ./datalake).")
    ap.add_argument("--min-chars", type=int, default=1, help="Descarta textos abaixo deste tamanho.")
    ap.add_argument("--drop-review", action="store_true", help="Exclui needs_human_review=true.")
    ap.add_argument("--shard-size", type=int, default=5000, help="Docs por shard .jsonl.zst.")
    ap.add_argument("--limite-split", type=int, default=LIMITE_LONGO,
                    help="Fatia docs acima deste nº de tokens (padrão 8192).")
    ap.add_argument("--no-near-dup", dest="near_dup", action="store_false",
                    help="Pula a deduplicação aproximada (D-1).")
    ap.add_argument("--threshold", type=float, default=0.85, help="Limiar Jaccard do near-dup.")
    ap.add_argument("--num-perm", type=int, default=128, help="Permutações MinHash.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )
    res = build_corpus(args.root, args.min_chars, args.drop_review, args.shard_size,
                       args.limite_split, args.near_dup, args.threshold, args.num_perm)
    print(
        f"\n  Limpo:               {res['limpo']}\n"
        f"  Base (filtrada):     {res['base']}\n"
        f"  D-2 fatiamento:      {res['fatiados']} docs → {res['fatias']} fatias  (pós: {res['pos_split']})\n"
        f"  D-1 near-dup remov.: {res['near_dup_removidos']}\n"
        f"  Classes de tamanho:  {res['tamanho_classe']}\n"
        f"  TRAIN (canônicos):   {res['train']}\n"
        f"  RAW (tudo):          {res['raw']}\n"
        f"  Tokens (train):      ~{res['tokens_estimados']:,}\n"
        f"  Shards .jsonl.zst:   {res['shards']}\n"
        f"  Saída:               {res['dir']}\n"
    )


if __name__ == "__main__":
    sys.exit(main())
