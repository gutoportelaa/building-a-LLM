#!/usr/bin/env python3
"""Publica no HuggingFace: modelo DAPT (privado) + índice RAG (dataset)."""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()
USER = "gutoportelaa"

MODEL_REPO = f"{USER}/qwen2.5-1.5b-dompi-dapt"
DS_REPO = f"{USER}/dompi-rag-index-e5"

MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
language: [pt]
tags: [dom-pi, dapt, qlora, piaui, government, continued-pretraining]
---

# Qwen2.5-1.5B — DAPT DOM-PI (QLoRA NF4, merged)

Pré-treino continuado (DAPT) de `Qwen/Qwen2.5-1.5B` sobre o corpus **DOM-PI**
(Diário Oficial dos Municípios do Piauí). Melhor checkpoint por CE no held-out
(QLoRA 4-bit NF4, merged). Usado como gerador **G2** na avaliação de RAG (Q5).

## Resultados
- Q1 (DAPT): degradação mínima de PPL no held-out geral (~+4,4%); aprendeu padrões
  estruturais de portarias/contratos. Corpus ~50 M tokens (~200× menor que o mínimo
  da literatura para DAPT efetivo).
- Q5 (RAG): acerto factual 0% (sem RAG) → 16,7% (com RAG standard), entre o base
  (11,1%) e o qwen2.5:14b (33,3%).

## Uso
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gutoportelaa/qwen2.5-1.5b-dompi-dapt")
model = AutoModelForCausalLM.from_pretrained("gutoportelaa/qwen2.5-1.5b-dompi-dapt")
```
"""

DS_CARD = """---
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
"""


def main():
    # ---- modelo DAPT (privado) ----
    print("Criando repo do modelo (privado)...", flush=True)
    create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
    Path("modelos/dapt_qlora_best/README.md").write_text(MODEL_CARD, encoding="utf-8")
    print("Upload do modelo DAPT...", flush=True)
    api.upload_folder(folder_path="modelos/dapt_qlora_best", repo_id=MODEL_REPO,
                      repo_type="model", commit_message="DAPT QLoRA NF4 merged (Q1 best)")
    print(f"OK modelo: https://huggingface.co/{MODEL_REPO}", flush=True)

    # ---- índice RAG (dataset, público por padrão) ----
    print("Criando repo do dataset...", flush=True)
    create_repo(DS_REPO, repo_type="dataset", private=True, exist_ok=True)
    Path("rag/index/README.md").write_text(DS_CARD, encoding="utf-8")
    print("Upload do índice RAG...", flush=True)
    api.upload_folder(folder_path="rag/index", repo_id=DS_REPO, repo_type="dataset",
                      commit_message="Índice e5-base 175.924 chunks DOM-PI")
    print(f"OK dataset: https://huggingface.co/datasets/{DS_REPO}", flush=True)


if __name__ == "__main__":
    main()
