#!/usr/bin/env python3
"""
importar_jsonl_chroma.py — Importa embeddings JSONL → ChromaDB
--------------------------------------------------------------
Roda DEPOIS que o construir_banco.py gerou o JSONL.
Requer ambiente sem processos competindo pelo ChromaDB.

Uso:
    uv run python src/vector_db/importar_jsonl_chroma.py \
        --jsonl dados/embeddings_marker.jsonl \
        --db-dir dados/chroma_db_marker
"""

import argparse
import json
import logging
import sys
import gc
from pathlib import Path

import chromadb

log = logging.getLogger("importar_chroma")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

BATCH_SIZE = 100  # ChromaDB suporta bem lotes de 100


def importar(jsonl_path: Path, db_dir: Path, collection_name: str) -> None:
    if not jsonl_path.exists():
        log.error(f"Arquivo não encontrado: {jsonl_path}")
        sys.exit(1)

    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    log.info(f"Importando {jsonl_path} → {db_dir}/{collection_name}")

    ids, docs, metas, embs = [], [], [], []
    total = 0
    skipped = 0

    # IDs já presentes (evita duplicatas em re-execução)
    existing_ids: set = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
        log.info(f"Coleção existente: {len(existing_ids)} chunks já presentes.")
    except Exception:
        pass

    def flush():
        nonlocal total
        if not ids:
            return
        try:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            total += len(ids)
            log.info(f"{total} chunks importados...")
        except Exception as e:
            log.error(f"Erro ao inserir lote: {e}")
        ids.clear(); docs.clear(); metas.clear(); embs.clear()
        gc.collect()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Linha {line_num} inválida: {e}")
                continue

            chunk_id = rec.get("id")
            if not chunk_id or not rec.get("embedding"):
                log.warning(f"Linha {line_num} sem id ou embedding, pulando.")
                continue

            if chunk_id in existing_ids:
                skipped += 1
                continue

            ids.append(chunk_id)
            docs.append(rec.get("document", ""))
            metas.append(rec.get("metadata", {}))
            embs.append(rec["embedding"])

            if len(ids) >= BATCH_SIZE:
                flush()

    flush()
    log.info(f"Importação concluída. {total} chunks inseridos | {skipped} já existiam.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--db-dir", default="dados/chroma_db_marker")
    parser.add_argument("--collection", default="dompi_documentos")
    args = parser.parse_args()

    importar(Path(args.jsonl), Path(args.db_dir), args.collection)


if __name__ == "__main__":
    main()
