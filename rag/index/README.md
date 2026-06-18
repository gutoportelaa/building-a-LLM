---
license: cc-by-4.0
language: [pt]
tags: [rag, dom-pi, embeddings, e5, faiss]
---

# DOM-PI RAG Index (multilingual-e5-base)

Índice vetorial do corpus DOM-PI para RAG (Questão 5).

- **175.924 chunks** de 57.229 documentos (janela 1600 chars, overlap 200)
- Embeddings `intfloat/multilingual-e5-base` (768-d, L2-normalizados)

## Arquivos
- `embeddings.npy` — matriz float32 N×768
- `chunks.jsonl` — `{chunk_id, doc_id, texto}`
- `meta.json` — configuração e estatísticas

Regenerável via `rag/build_index.py`. Prefixos e5: `query:` / `passage:`.
