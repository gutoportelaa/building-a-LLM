#!/usr/bin/env python3
"""Publica no HuggingFace os modelos da Questão 1 (rodar NO CLUSTER, onde estão
o token HF e os checkpoints):
  - Full FT unificado (resposta canônica ao enunciado: dataset COMPLETO DOM-PI; degradado)
  - Full FT Teresina v3 (alternativa: corpus curado + objetivo de treino corrigido; ganho real)
Ambos privados."""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()
USER = "gutoportelaa"

UNIF_REPO = f"{USER}/qwen2.5-1.5b-dompi-fullft-unificado"
V3_REPO = f"{USER}/qwen2.5-1.5b-dompi-teresina-v3"
UNIF_CKPT = "treino/checkpoints_fullft_unificado/best"
V3_CKPT = "treino/checkpoints_fullft_teresina_v3/best"

# Bloco comum às duas cards: tabela comparativa + bug + exemplo de inferência.
COMUM = """
## Contexto — Questão 1 (Pré-treino Continuado / DAPT)

Pré-treino continuado de `Qwen/Qwen2.5-1.5B` sobre o **DOM-PI** (Diário Oficial dos
Municípios do Piauí). Foram exploradas várias implementações; a tabela resume o held-out:

| Implementação | Corpus | held-out PPL | Δ vs baseline |
|---|---|---|---|
| Baseline Qwen2.5-1.5B | — | 6,91 (Teresina) / 10,03 (unif) | — |
| **Full FT — dataset COMPLETO** | DOM-PI unificado | 22,22 | **+121% (degradado)** |
| QLoRA | DOM-PI unificado | 10,47 | +4,4% (PEFT mais robusto) |
| LoRA | DOM-PI unificado | ~1.000 | colapso (intruder dimensions) |
| Full FT Teresina v1 (1 ép) | Teresina curado | 8,76 | +27% |
| Full FT Teresina v2 (3 ép + freeze) | Teresina curado | 12,64 | +83% |
| **Full FT Teresina v3 (objetivo corrigido)** | Teresina curado | **6,02** | **−12,9% (ganho real)** |

### A correção decisiva (bug do duplo deslocamento)
As rodadas v1/v2/unificado treinaram com um **duplo shift de rótulos**: o dataset
pré-deslocava os `labels` (`block[1:]`) e o modelo HuggingFace também desloca
internamente, fazendo o treino otimizar "prever o token 2 posições à frente". Loss
de treino ~11 e PPL interna ~47.000. Corrigido o alinhamento (`input_ids == labels`),
o **v3** produziu o primeiro ganho real. Métricas v3: held-out PPL 6,91→6,02,
token-acc 58,8%→60,8%; benchmark PPL 7,45→7,24.

### Exemplo de inferência (completar portaria)
Prompt: *"PORTARIA Nº 130… O PREFEITO MUNICIPAL DE COCAL… RESOLVE: Art. 1º"*
- **Full FT unificado (degradado):** "…limpeza urbana e *coletares resídeos* domiciliares… *Coletes Resídeos R$ 70,00*" — gramática/OCR degradados.
- **Full FT Teresina v3:** "Fica autorizado o pagamento da folha de pessoal dos servidores municipais… conforme Lei nº 4.679/2025, em regime de 40 horas…" — fluente e no domínio.

Avaliação como gerador **G2** no sistema de RAG (Questão 5).
"""

CARD_UNIF = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
language: [pt]
tags: [dom-pi, dapt, full-finetuning, piaui, government, continued-pretraining]
---

# Qwen2.5-1.5B — DAPT DOM-PI (Full FT, dataset COMPLETO)

**Resposta canônica à Questão 1**: pré-treino continuado de parâmetros plenos sobre o
**dataset completo** do DOM-PI (corpus unificado, 224 municípios + capital, OCR ruidoso).
Apresentado **mesmo degradado** (+121% PPL) por ser exatamente o que o enunciado pede
(usar o dataset completo) e por ilustrar a dificuldade do full FT sobre corpus ruidoso.
A alternativa que resolve o problema é [`{V3_REPO}`](https://huggingface.co/{V3_REPO}).
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
**subcorpus curado de Teresina** (tier A+B, ~9,3 M tokens) com o **objetivo de treino
corrigido**. É a tentativa bem-sucedida de resolver a degradação observada no pré-treino
com o dataset completo: **ganho real de −12,9% PPL no held-out**. A resposta canônica
(dataset completo, degradada) é [`{UNIF_REPO}`](https://huggingface.co/{UNIF_REPO}).
{COMUM}
"""


def upload(repo, ckpt, card):
    print(f"--- {repo}", flush=True)
    create_repo(repo, repo_type="model", private=True, exist_ok=True)
    Path(ckpt, "README.md").write_text(card, encoding="utf-8")
    api.upload_folder(folder_path=ckpt, repo_id=repo, repo_type="model",
                      commit_message="Q1 DAPT checkpoint + model card")
    print(f"OK: https://huggingface.co/{repo}", flush=True)


if __name__ == "__main__":
    upload(UNIF_REPO, UNIF_CKPT, CARD_UNIF)
    upload(V3_REPO, V3_CKPT, CARD_V3)
    print("CONCLUÍDO.", flush=True)
