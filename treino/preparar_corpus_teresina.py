#!/usr/bin/env python3
"""
preparar_corpus_teresina.py — Baixa corpus Teresina do HuggingFace e salva como JSONL.

Usa a configuração 'curated' do dataset (9.577 docs, tier A+B) para maximizar
qualidade. Exclui do treino os docs já presentes no held-out local (por ID).

Dataset: gutoportelaa/dom-pi-teresina-2025

Saídas:
  data/teresina_hf/train_corpus.jsonl — corpus de treino (tier A+B, excl. held-out)
  data/teresina_hf/stats.json          — estatísticas do corpus

Uso:
  .venv/bin/python3 treino/preparar_corpus_teresina.py \
      --output-dir data/teresina_hf \
      --held-out data/teresina/held_out.jsonl \
      --quality-tiers A B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/teresina_hf")
    p.add_argument("--held-out", default="data/teresina/held_out.jsonl")
    p.add_argument("--quality-tiers", nargs="+", default=["A", "B"])
    p.add_argument("--dataset-name", default="gutoportelaa/dom-pi-teresina-2025")
    p.add_argument("--config", default="curated", help="Configuração HF do dataset")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Instalando datasets...", flush=True)
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "--quiet"], check=True)
        from datasets import load_dataset

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carrega IDs do held-out para excluir do treino
    held_out_ids: set[str] = set()
    held_out_path = Path(args.held_out)
    if held_out_path.exists():
        with open(held_out_path) as f:
            for line in f:
                obj = json.loads(line)
                doc_id = str(obj.get("id") or obj.get("_id") or "")
                if doc_id:
                    held_out_ids.add(doc_id)
        print(f"Held-out: {len(held_out_ids)} IDs a excluir do treino", flush=True)
    else:
        print(f"[aviso] held-out não encontrado em {args.held_out} — usando todos os docs", flush=True)

    print(f"Baixando {args.dataset_name} (config={args.config})...", flush=True)
    try:
        ds = load_dataset(args.dataset_name, args.config, split="train", trust_remote_code=True)
    except Exception:
        print(f"Config '{args.config}' falhou, tentando config padrão...", flush=True)
        ds = load_dataset(args.dataset_name, split="train", trust_remote_code=True)

    print(f"Dataset carregado: {len(ds)} docs | Colunas: {ds.column_names}", flush=True)

    train_path = out_dir / "train_corpus.jsonl"
    n_included = 0
    n_skip_quality = 0
    n_skip_held_out = 0
    total_chars = 0

    with open(train_path, "w", encoding="utf-8") as f:
        for row in ds:
            quality = str(row.get("quality_tier", "A"))
            if quality not in args.quality_tiers:
                n_skip_quality += 1
                continue

            doc_id = str(row.get("id", ""))
            if doc_id and doc_id in held_out_ids:
                n_skip_held_out += 1
                continue

            texto = row.get("texto") or row.get("text") or ""
            if len(texto) < 50:
                continue

            record = {
                "id": doc_id,
                "municipio": row.get("municipio", ""),
                "tipo_ato": row.get("tipo_ato", ""),
                "ano": row.get("ano", ""),
                "quality_tier": quality,
                "texto": texto,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_included += 1
            total_chars += len(texto)

    n_tokens_est = total_chars // 4  # ~4 chars/token para português

    stats = {
        "dataset": args.dataset_name,
        "config": args.config,
        "n_docs_total": len(ds),
        "n_docs_treino": n_included,
        "n_skip_quality": n_skip_quality,
        "n_skip_held_out": n_skip_held_out,
        "total_chars": total_chars,
        "n_tokens_estimado": n_tokens_est,
        "quality_tiers": args.quality_tiers,
        "output": str(train_path),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nResultado:")
    print(f"  Docs treino:         {n_included:,}")
    print(f"  Skip (qualidade):    {n_skip_quality:,}")
    print(f"  Skip (held-out):     {n_skip_held_out:,}")
    print(f"  Total caracteres:    {total_chars:,}")
    print(f"  Tokens estimados:    {n_tokens_est:,}")
    print(f"  Corpus salvo em:     {train_path}")
    print(f"  Stats:               {out_dir/'stats.json'}", flush=True)


if __name__ == "__main__":
    main()
