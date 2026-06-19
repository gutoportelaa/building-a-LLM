#!/usr/bin/env python3
"""Publica no HuggingFace os modelos da Questão 1 (rodar NO CLUSTER, onde estão
o token HF e os checkpoints):
  - Full FT unificado v2 (resposta canônica ao enunciado: dataset COMPLETO DOM-PI,
    objetivo de treino corrigido; GANHO real −11,3%)
  - Full FT Teresina v3 (alternativa: corpus curado + objetivo corrigido; −12,9%)
Ambos privados."""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()
USER = "gutoportelaa"

UNIF_REPO = f"{USER}/qwen2.5-1.5b-dompi-fullft-unificado"
V3_REPO = f"{USER}/qwen2.5-1.5b-dompi-teresina-v3"
UNIF_CKPT = "treino/checkpoints_fullft_unificado_v2/best"   # v2 corrigido (substitui o degradado)
V3_CKPT = "treino/checkpoints_fullft_teresina_v3/best"

# Bloco comum às duas cards: tabela comparativa + bug + exemplo de inferência.
COMUM = """
## Contexto — Questão 1 (Pré-treino Continuado / DAPT)

Pré-treino continuado de `Qwen/Qwen2.5-1.5B` sobre o **DOM-PI** (Diário Oficial dos
Municípios do Piauí). Resultados no held-out, **com o objetivo de treino corrigido**:

| Implementação | Corpus | held-out PPL | Δ vs baseline |
|---|---|---|---|
| Baseline Qwen2.5-1.5B | — | 10,03 (unif) / 6,91 (Teresina) | — |
| **Full FT — dataset COMPLETO (v2)** | DOM-PI unificado | **8,90** | **−11,3% ✓** |
| **Full FT Teresina v3** | Teresina curado | **6,02** | **−12,9% ✓** |
| QLoRA — dataset completo (v2) | DOM-PI unificado | 9,88 | −1,4% |

Em ambos os corpora, o **Full FT de parâmetros plenos supera o PEFT** (QLoRA) na
adaptação de domínio — como a teoria de DAPT prevê.

### O bug que invertia a conclusão (duplo deslocamento)
As primeiras rodadas treinaram com um **duplo shift de rótulos**: o dataset
pré-deslocava os `labels` (`block[1:]`) e o modelo HuggingFace também desloca
internamente, fazendo o treino otimizar "prever o token 2 posições à frente". Isso
inflava a loss de treino para ~11 e dava resultados degradados (full FT unificado
parecia +121%, e o QLoRA "vencia" com +4,4%). Corrigido o alinhamento
(`input_ids == labels`), o full FT unificado foi de **+121% → −11,3%** e passou a
superar o PEFT — invertendo a conclusão obtida sob o bug.

### Exemplo de inferência (completar portaria)
Prompt: *"PORTARIA Nº 130… O PREFEITO MUNICIPAL DE COCAL… RESOLVE: Art. 1º"*
- **Bugado (full FT unificado):** "…limpeza urbana e *coletares resídeos* domiciliares… *Coletes Resídeos R$ 70,00*" — gramática/OCR degradados.
- **Corrigido (full FT unificado v2):** "O Diário Oficial dos Municípios do Piauí (DOMP) é um periódico oficial produzido pela Secretaria Municipal… publicado diariamente…" — fluente e no domínio.

Avaliação como gerador **G2** no sistema de RAG (Questão 5).
"""

CARD_UNIF = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
language: [pt]
tags: [dom-pi, dapt, full-finetuning, piaui, government, continued-pretraining]
---

# Qwen2.5-1.5B — DAPT DOM-PI (Full FT, dataset COMPLETO, objetivo corrigido)

**Resposta canônica à Questão 1**: pré-treino continuado de parâmetros plenos sobre o
**dataset completo** do DOM-PI (corpus unificado, 224 municípios + capital). Com o objetivo
de treino corrigido, **melhora o modelo em −11,3%** no held-out (PPL 10,03 → 8,90; token-acc
54,1% → 55,9%; benchmark 7,45 → 7,26) — é a resposta literal ao enunciado e o melhor resultado
no held-out geral. Resposta alternativa (corpus curado): [`{V3_REPO}`](https://huggingface.co/{V3_REPO}).
{COMUM}
"""

CARD_V3 = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
language: [pt]
tags: [dom-pi, dapt, full-finetuning, piaui, government, continued-pretraining]
---

# Qwen2.5-1.5B — DAPT DOM-PI Teresina v3 (objetivo corrigido)

**Resposta alternativa à Questão 1**: pré-treino continuado de parâmetros plenos sobre o
**subcorpus curado de Teresina** (tier A+B, ~9,3 M tokens) com o **objetivo de treino corrigido**.
Ganho de **−12,9% PPL** no held-out de Teresina (6,91 → 6,02; token-acc 58,8% → 60,8%). A resposta
canônica (dataset completo) é [`{UNIF_REPO}`](https://huggingface.co/{UNIF_REPO}).
{COMUM}
"""


def upload(repo, ckpt, card):
    print(f"--- {repo}", flush=True)
    create_repo(repo, repo_type="model", private=True, exist_ok=True)
    Path(ckpt, "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(folder_path=ckpt, repo_id=repo, repo_type="model",
                      commit_message="Q1 DAPT v2 (objetivo corrigido) + model card")
    print(f"OK: https://huggingface.co/{repo}", flush=True)


if __name__ == "__main__":
    upload(UNIF_REPO, UNIF_CKPT, CARD_UNIF)
    upload(V3_REPO, V3_CKPT, CARD_V3)
    print("CONCLUÍDO.", flush=True)
