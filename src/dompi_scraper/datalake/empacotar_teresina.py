#!/usr/bin/env python3
"""
empacotar_teresina.py — dataset ISOLADO da capital (Teresina), self-contained.

Teresina publica em diário próprio (DOM-Teresina), fora do DOM-PI dos Municípios. Por
ser a capital, além de entrar no dataset geral como 13º território, ganha um repositório
isolado com **texto + PDFs no mesmo repo** (é pequeno, ~1,1 GB — não há o problema de
volume do geral). Filtra a camada corpus por `territorio == teresina` e gera:

  hf_teresina/
    data/train-*.parquet      # config default (limpo+tier+dedup)
    curated/curated-*.parquet # Tier A+B
    raw/raw-*.parquet         # tudo + flags
    README.md                 # dataset card da capital
    (pdfs/ é subido à parte, de territorios/teresina/pdfs)

Uso:
    python -m dompi_scraper.datalake.empacotar_teresina
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from . import zone_dir
from .empacotar_hf import _write_parquet_chunks

log = logging.getLogger("empacotar_teresina")
REPO = "gutoportelaa/dom-pi-teresina-2025"
SLUG = "teresina"


def _readme(n_train: int, n_curated: int, n_raw: int, tokens: int, tipo: list[dict],
            tier: dict) -> str:
    tipo_list = ", ".join(f"{d['tipo_ato']} ({d['count']})" for d in tipo)
    return f"""---
license: cc-by-4.0
language:
  - pt
pretty_name: "DOM-Teresina 2025 — Diário Oficial do Município de Teresina (PI)"
task_categories:
  - text-generation
  - fill-mask
tags:
  - legal
  - government
  - brazil
  - piaui
  - teresina
  - official-gazette
  - portuguese
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
  - config_name: curated
    data_files:
      - split: train
        path: curated/curated-*.parquet
  - config_name: raw
    data_files:
      - split: raw
        path: raw/raw-*.parquet
---

# DOM-Teresina 2025 — Diário Oficial do Município de Teresina (PI)

Texto integral das publicações de **2025** do **Diário Oficial do Município de Teresina**
(DOM-Teresina), a **capital do Piauí** — que publica em diário próprio, separado do DOM-PI
dos Municípios. **{n_train} documentos** · ~{tokens:,} tokens · pt-BR.

> Parte da capital também está no dataset geral dos demais municípios:
> [`gutoportelaa/dom-pi-corpus-2025`](https://huggingface.co/datasets/gutoportelaa/dom-pi-corpus-2025)
> (como território `teresina`). Este repositório é **self-contained**: inclui os **PDFs-fonte**
> em `pdfs/`.

## Configurações

| Config | Split | Conteúdo |
|---|---|---|
| `default` | `train` | {n_train} docs — limpeza v2 + `quality_tier` + dedup. |
| `curated` | `train` | {n_curated} docs — Tier A+B (prosa aproveitável). |
| `raw` | `raw` | {n_raw} docs — sem remover near-dups, com `cluster_id`. |

```python
from datasets import load_dataset
ds = load_dataset("{REPO}", split="train")
```

## Esquema

`id`, `municipio` (=Teresina), `tipo_ato`, `ano`, `data_publicacao`, `n_tokens`,
`tamanho_classe`, `quality_tier`, `texto`. Tipos mais comuns: {tipo_list}.

Tiers de qualidade: A **{tier.get('A',0)}** · B **{tier.get('B',0)}** · C **{tier.get('C',0)}**
(C = tabela fiscal achatada / ruído de OCR — caso de re-extração).

## Fonte e licença

- **Fonte:** Diário Oficial do Município de Teresina — publicações oficiais de 2025 (documentos públicos).
- **PDFs-fonte:** incluídos em `pdfs/` neste mesmo repositório.
- **Licença:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) — atribua à fonte (DOM-Teresina).
""".replace(",", ".")


def empacotar(root=None, out: str = "hf_teresina", rows_per_file: int = 50000) -> dict:
    corpus_dir = zone_dir("corpus", root)
    train = pl.read_parquet(str(corpus_dir / "corpus_llm" / "**" / "*.parquet"),
                            hive_partitioning=True).filter(pl.col("territorio") == SLUG)
    raw = pl.read_parquet(str(corpus_dir / "corpus_raw" / "**" / "*.parquet"),
                          hive_partitioning=True).filter(pl.col("territorio") == SLUG)
    if train.is_empty():
        raise SystemExit("Nenhum doc de teresina no corpus. Rode build_corpus após ingerir teresina.")
    cols = ["id", "municipio", "tipo_ato", "ano", "data_publicacao",
            "n_tokens", "tamanho_classe", "quality_tier", "texto"]
    train = train.select([c for c in cols if c in train.columns])
    raw_cols = cols + [c for c in ("cluster_id", "is_near_dup") if c in raw.columns]
    raw = raw.select([c for c in raw_cols if c in raw.columns])
    curated = train.filter(pl.col("quality_tier").is_in(["A", "B"])) \
        if "quality_tier" in train.columns else train

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_chunks(train, out_dir / "data", "train", rows_per_file)
    _write_parquet_chunks(curated, out_dir / "curated", "curated", rows_per_file)
    _write_parquet_chunks(raw, out_dir / "raw", "raw", rows_per_file)

    tipo = train.group_by("tipo_ato").agg(pl.len().alias("count")).sort("count", descending=True).head(6).to_dicts()
    tier = {d["quality_tier"]: d["k"] for d in
            train.group_by("quality_tier").agg(pl.len().alias("k")).to_dicts()} \
        if "quality_tier" in train.columns else {}
    (out_dir / "README.md").write_text(
        _readme(train.height, curated.height, raw.height, int(train["n_tokens"].sum()), tipo, tier),
        encoding="utf-8")
    log.info("Teresina empacotado: train=%d curated=%d raw=%d → %s/", train.height, curated.height, raw.height, out)
    return {"train": train.height, "curated": curated.height, "raw": raw.height,
            "tokens": int(train["n_tokens"].sum()), "tier": tier, "out": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Empacota o dataset isolado de Teresina (capital).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="hf_teresina")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    s = empacotar(args.root, args.out)
    print(f"\n  TRAIN {s['train']} · CURATED {s['curated']} · RAW {s['raw']} · ~{s['tokens']:,} tokens"
          f"\n  Tiers: {s['tier']}\n  Saída: {s['out']}/  (data/ curated/ raw/ README.md)"
          f"\n\n  Upload (texto + PDFs no MESMO repo):"
          f"\n    hf upload {REPO} {s['out']} . --repo-type=dataset"
          f"\n    hf upload {REPO} territorios/teresina/pdfs pdfs --repo-type=dataset")


if __name__ == "__main__":
    sys.exit(main())
