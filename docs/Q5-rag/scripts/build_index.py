#!/usr/bin/env python3
"""
build_index.py — Constrói o índice vetorial do corpus DOM-PI para RAG.

Pipeline:
  1. Lê os documentos de corpus (train_corpus + held_out) — campos {id, texto}.
  2. Divide cada documento em chunks de ~CHUNK_CHARS caracteres com sobreposição.
  3. Gera embeddings com intfloat/multilingual-e5-base na GPU (prefixo "passage: ").
  4. Salva:
       index/embeddings.npy  — matriz float32 N×768 já L2-normalizada
       index/chunks.jsonl    — {chunk_id, doc_id, texto} (texto SEM prefixo e5)
       index/meta.json       — configuração e estatísticas

Decisão de projeto: o fonte_id do benchmark NÃO casa com os ids do corpus
(ligação quebrada na origem), então a avaliação de recuperação é feita por
CONTEÚDO (ver run_eval.py), não por id. O índice cobre todo o corpus local
para que os documentos-fonte das respostas estejam presentes.

Uso:
  .venv/bin/python3 rag/build_index.py \
      --corpus data/train_corpus.jsonl data/held_out.jsonl \
      --out-dir rag/index
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def iter_chunks(texto: str, chunk_chars: int, overlap: int):
    """Divide um texto em janelas de chunk_chars com sobreposição de overlap."""
    texto = texto.strip()
    if len(texto) <= chunk_chars:
        if texto:
            yield texto
        return
    step = chunk_chars - overlap
    for start in range(0, len(texto), step):
        piece = texto[start:start + chunk_chars].strip()
        if len(piece) >= 64:  # descarta fragmentos minúsculos
            yield piece
        if start + chunk_chars >= len(texto):
            break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", nargs="+",
                   default=["data/train_corpus.jsonl", "data/held_out.jsonl"])
    p.add_argument("--out-dir", default="rag/index")
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--chunk-chars", type=int, default=1600)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-docs", type=int, default=0, help="0 = todos")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1+2. lê corpus e gera chunks
    chunks = []          # texto puro
    meta = []            # (chunk_id, doc_id)
    n_docs = 0
    for fp in args.corpus:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = r.get("id") or r.get("_id") or f"doc{n_docs}"
                texto = r.get("texto") or r.get("text") or ""
                if not texto.strip():
                    continue
                for ci, piece in enumerate(iter_chunks(texto, args.chunk_chars, args.overlap)):
                    meta.append((f"{doc_id}::c{ci}", str(doc_id)))
                    chunks.append(piece)
                n_docs += 1
                if args.max_docs and n_docs >= args.max_docs:
                    break
        if args.max_docs and n_docs >= args.max_docs:
            break

    print(f"Documentos lidos: {n_docs:,} | chunks: {len(chunks):,}", flush=True)

    # 3. embeddings na GPU
    from sentence_transformers import SentenceTransformer
    print(f"Carregando {args.model}...", flush=True)
    model = SentenceTransformer(args.model, device="cuda")

    t0 = time.time()
    # e5 exige prefixo "passage: " nos documentos
    passages = [f"passage: {c}" for c in chunks]
    emb = model.encode(
        passages, batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")
    dt = time.time() - t0
    print(f"Embeddings: {emb.shape} em {dt:.0f}s ({len(chunks)/dt:.0f} emb/s)", flush=True)

    # 4. salva
    np.save(out / "embeddings.npy", emb)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for (chunk_id, doc_id), texto in zip(meta, chunks):
            f.write(json.dumps({"chunk_id": chunk_id, "doc_id": doc_id, "texto": texto},
                               ensure_ascii=False) + "\n")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "dim": int(emb.shape[1]),
            "n_docs": n_docs, "n_chunks": len(chunks),
            "chunk_chars": args.chunk_chars, "overlap": args.overlap,
            "corpus": args.corpus,
        }, f, indent=2, ensure_ascii=False)
    print(f"Índice salvo em {out}/ (embeddings.npy, chunks.jsonl, meta.json)", flush=True)


if __name__ == "__main__":
    main()
