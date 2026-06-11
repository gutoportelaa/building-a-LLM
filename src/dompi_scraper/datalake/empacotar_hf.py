#!/usr/bin/env python3
"""
empacotar_hf.py — empacota o corpus para publicação no HuggingFace (reproduzível).

Lê os produtos do data lake (corpus/corpus_llm = train; corpus/corpus_raw = raw) e
monta o diretório `hf_corpus_dompi/`:

  hf_corpus_dompi/
    data/train-*.parquet        # config 'default', split 'train' (canônicos, dedup)
    raw/raw-*.parquet           # config 'raw', split 'raw' (tudo + cluster_id/flags)
    shards/part-*.jsonl.zst     # shards de treino (copiados do corpus_llm)
    README.md                   # dataset card com estatísticas calculadas

NÃO faz upload — isso é um passo manual explícito:
    hf upload-large-folder gutoportelaa/dom-pi-corpus-2025 hf_corpus_dompi --repo-type=dataset

Uso:
    python -m dompi_scraper.datalake.empacotar_hf
    python -m dompi_scraper.datalake.empacotar_hf --rows-per-file 40000
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import polars as pl

from . import zone_dir

log = logging.getLogger("empacotar_hf")

REPO = "gutoportelaa/dom-pi-corpus-2025"


def _write_parquet_chunks(df: pl.DataFrame, dest_dir: Path, stem: str, rows_per_file: int) -> int:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, -(-df.height // rows_per_file))  # ceil
    for i in range(n):
        df.slice(i * rows_per_file, rows_per_file).write_parquet(
            dest_dir / f"{stem}-{i:05d}-of-{n:05d}.parquet", compression="zstd")
    return n


def _stats(train: pl.DataFrame, raw: pl.DataFrame) -> dict:
    terr = (train.group_by("territorio").agg(pl.len().alias("k")).sort("k", descending=True))
    tipo = (train.group_by("tipo_ato").agg(pl.len().alias("k")).sort("k", descending=True).head(8))
    classe = (raw.group_by("tamanho_classe").agg(pl.len().alias("k")))
    tier = {d["quality_tier"]: d["k"] for d in
            train.group_by("quality_tier").agg(pl.len().alias("k")).to_dicts()} \
        if "quality_tier" in train.columns else {}
    return {
        "n_train": train.height,
        "n_raw": raw.height,
        "n_curated": int(train["quality_tier"].is_in(["A", "B"]).sum()) if tier else train.height,
        "n_near_dup": raw.height - train.height,
        "tokens": int(train["n_tokens"].sum()),
        "n_munis": train.filter(pl.col("municipio") != "DESCONHECIDO")["municipio"].n_unique(),
        "n_desconhecido": int((train["municipio"] == "DESCONHECIDO").sum()),
        "terr": terr.to_dicts(),
        "tipo": tipo.to_dicts(),
        "classe": {d["tamanho_classe"]: d["k"] for d in classe.to_dicts()},
        "tier": tier,
    }


def _readme(s: dict) -> str:
    terr_rows = "\n".join(f"| {d['territorio']} | {d['k']:,} |" for d in s["terr"])
    tipo_list = ", ".join(f"{d['tipo_ato']} ({d['k']:,})" for d in s["tipo"])
    cl = s["classe"]
    body = f"""---
license: cc-by-4.0
language:
  - pt
pretty_name: "Corpus DOM-PI 2025 — Diário Oficial dos Municípios do Piauí"
task_categories:
  - text-generation
  - fill-mask
tags:
  - legal
  - government
  - brazil
  - piaui
  - official-gazette
  - portuguese
  - ocr
size_categories:
  - 10K<n<100K
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

# Corpus DOM-PI 2025 — Diário Oficial dos Municípios do Piauí

Texto integral das publicações de **2025** do **Diário Oficial dos Municípios do
Piauí (DOM-PI)**, extraído de PDFs por OCR/parsing e organizado por **Território de
Desenvolvimento** e **município**. Config `default` (split `train`): **{s['n_train']:,}
documentos** · **~{s['tokens']//1_000_000} milhões de tokens** · **12 territórios** · pt-BR.

> *Full text of the 2025 official gazettes of the municipalities of Piauí (Brazil),
> OCR/parsed from PDFs, organized by development territory and municipality.*

## Configurações (configs)

| Config | Split | Conteúdo |
|---|---|---|
| `default` | `train` | {s['n_train']:,} docs **deduplicados** (exato + quase-duplicatas), longos **fatiados em atos**, com **limpeza v2** (boilerplate removido). Pré-treino. |
| `curated` | `train` | {s['n_curated']:,} docs **Tier A+B** (prosa aproveitável; exclui tabela fiscal achatada). Base para SFT/instruction. |
| `raw` | `raw` | {s['n_raw']:,} docs (mesmo pós-processamento) **sem remover quase-duplicatas**, com `cluster_id`/`is_near_dup` para auditoria ou dedup própria. |

```python
from datasets import load_dataset
ds  = load_dataset("{REPO}", split="train")                 # default (pré-treino)
cur = load_dataset("{REPO}", "curated", split="train")      # Tier A+B (SFT)
raw = load_dataset("{REPO}", "raw", split="raw")            # tudo + flags de near-dup
ds = ds.filter(lambda r: r["territorio"] == "cocais")       # filtrar por território
```

## Esquema

Cada linha é **um ato/página publicado** (portaria, decreto, licitação, lei, etc.).

| Coluna | Descrição |
|---|---|
| `id` | MD5 do conteúdo normalizado (chave de dedup) |
| `territorio` | Território de Desenvolvimento (slug) |
| `municipio` | nome oficial canonizado ({s['n_munis']} valores; `DESCONHECIDO` quando irrecuperável) |
| `tipo_ato` | Portaria, Decreto, Licitação, Lei, LRF, Edital, Contrato… |
| `ano` | ano de publicação (2025) |
| `data_publicacao` | `DD/MM/AAAA` (maioria) ou `2025` (sem data precisa) |
| `n_tokens` | estimativa (~`n_chars/4`) |
| `tamanho_classe` | `normal` (≤8k tok) · `longo` (8k–32k) · `mega` (>32k, doc único não-fatiável) |
| `quality_tier` | `A` prosa limpa · `B` média · `C` tabela fiscal achatada/ruído (re-extração) |
| `texto` | texto limpo (limpeza v2: boilerplate/cabeçalho/assinatura removidos) |
| `cluster_id`, `is_near_dup` | *(só no config `raw`)* grupo de quase-duplicatas |

## Processamento de qualidade (D-1 / D-2)

- **Deduplicação exata** por hash de conteúdo normalizado (pré/pós-limpeza, cross-território).
- **Quase-duplicatas (D-1):** MinHash + LSH (Jaccard ≥ 0,85) agrupa near-dups (variações de
  OCR/rodapé); o `train` mantém 1 canônico por cluster — **{s['n_near_dup']:,} redundantes
  removidos**. O `raw` preserva todos com `cluster_id`.
- **Documentos longos (D-2):** docs > 8k tokens que são **compilações** de vários atos são
  **fatiados em atos atômicos** (fronteira = título de ato: PORTARIA/DECRETO/LEI/… Nº),
  preservando tabelas. Documentos únicos longos (orçamento, planilha, uma lei) ficam
  intactos e marcados `tamanho_classe`. Distribuição: normal **{cl.get('normal',0):,}** ·
  longo **{cl.get('longo',0):,}** · mega **{cl.get('mega',0):,}**.

## Estatísticas (config `default`)

**Por território:**

| Território | docs |
|---|---|
{terr_rows}

**Tipos de ato mais comuns:** {tipo_list}.

## Limitações conhecidas

- **Cobertura:** **Teresina** e **Parnaíba** não estão incluídas (diários próprios). 12 dos 13 Territórios.
- **OCR:** há ruído residual (cabeçalhos/assinaturas) em parte das páginas escaneadas.
- **Datas:** maioria com `DD/MM/AAAA`; uma fração tem apenas o ano (`2025`).
- **Município:** canonizado para a forma oficial; `DESCONHECIDO` (~{100*s['n_desconhecido']//max(s['n_train'],1)}%) para PDFs multi-município ou OCR irrecuperável.
- **Quase-duplicatas:** o `train` remove near-dups por similaridade; quase-duplicatas abaixo do limiar podem persistir.

## Fonte, licença e atribuição

- **Fonte:** Diário Oficial dos Municípios do Piauí — publicações oficiais de 2025.
- **Licença:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Atribua à fonte (DOM-PI / municípios do Piauí).
- Os textos são **atos oficiais públicos**; este dataset é uma compilação derivada para pesquisa/PLN.
"""
    # vírgulas de milhar (entre dígitos) → ponto, sem tocar vírgulas de prosa
    return re.sub(r"(?<=\d),(?=\d)", ".", body)


def empacotar(root=None, rows_per_file: int = 50000, out: str = "hf_corpus_dompi") -> dict:
    corpus_dir = zone_dir("corpus", root)
    # hive_partitioning recupera a coluna de partição `ano` (gravada só no caminho).
    train = pl.read_parquet(str(corpus_dir / "corpus_llm" / "**" / "*.parquet"),
                            hive_partitioning=True)
    raw = pl.read_parquet(str(corpus_dir / "corpus_raw" / "**" / "*.parquet"),
                          hive_partitioning=True)
    # garante ordem de colunas estável (ano vem da partição → reordena)
    train_cols = ["id", "territorio", "municipio", "tipo_ato", "ano",
                  "data_publicacao", "n_tokens", "tamanho_classe", "quality_tier", "texto"]
    train = train.select([c for c in train_cols if c in train.columns])
    # config 'curated': prosa aproveitável (Tier A+B), sem tabela achatada (C)
    curated = train.filter(pl.col("quality_tier").is_in(["A", "B"])) \
        if "quality_tier" in train.columns else train

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train_files = _write_parquet_chunks(train, out_dir / "data", "train", rows_per_file)
    n_cur_files = _write_parquet_chunks(curated, out_dir / "curated", "curated", rows_per_file)
    n_raw_files = _write_parquet_chunks(raw, out_dir / "raw", "raw", rows_per_file)
    # shards de treino: copia do corpus_llm/shards
    src_shards = corpus_dir / "corpus_llm" / "shards"
    dst_shards = out_dir / "shards"
    if dst_shards.exists():
        shutil.rmtree(dst_shards)
    if src_shards.exists():
        shutil.copytree(src_shards, dst_shards)

    s = _stats(train, raw)
    (out_dir / "README.md").write_text(_readme(s), encoding="utf-8")
    log.info("HF empacotado em %s/ (train=%d/%darq, curated=%d/%darq, raw=%d/%darq)",
             out, s["n_train"], n_train_files, s["n_curated"], n_cur_files,
             s["n_raw"], n_raw_files)
    s.update(train_files=n_train_files, curated_files=n_cur_files,
             raw_files=n_raw_files, out=str(out_dir))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Empacota o corpus para o HuggingFace (sem upload).")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="hf_corpus_dompi")
    ap.add_argument("--rows-per-file", type=int, default=50000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    s = empacotar(args.root, args.rows_per_file, args.out)
    print(
        f"\n  TRAIN:   {s['n_train']:,} docs → {s['train_files']} parquet  (~{s['tokens']:,} tokens)\n"
        f"  CURATED: {s['n_curated']:,} docs (Tier A+B) → {s['curated_files']} parquet\n"
        f"  RAW:     {s['n_raw']:,} docs → {s['raw_files']} parquet  (near-dup removidos: {s['n_near_dup']:,})\n"
        f"  Tiers: {s['tier']}  |  Classes: {s['classe']}\n"
        f"  Municípios: {s['n_munis']} (+ DESCONHECIDO: {s['n_desconhecido']:,})\n"
        f"  Saída: {s['out']}/  (data/ raw/ shards/ README.md)\n"
        f"\n  Upload (manual):\n"
        f"    hf upload-large-folder {REPO} {s['out']} --repo-type=dataset\n"
    )


if __name__ == "__main__":
    sys.exit(main())
