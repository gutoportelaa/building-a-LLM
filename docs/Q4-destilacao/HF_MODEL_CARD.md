---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
language: [pt]
tags: [knowledge-distillation, logit-kd, dom-pi, qwen2.5, distillation]
pipeline_tag: text-generation
---

# qwen2.5-1.5b-dompi-distill-combined

Aluno destilado da **Questão 4** do trabalho final (Construindo um LLM, UFPI-DC). Melhor configuração da matriz
de 12 alunos: **aluno 1.5B · professor com RAG (braço B) · método combinado (CE+KL)**.

## Como foi treinado (white-box logit KD, mesma família)
- **Professor:** `Qwen/Qwen2.5-14B-Instruct` (mesmo tokenizador → permite destilação por logits).
- **Aluno:** `Qwen/Qwen2.5-1.5B` (base pristino).
- **Dataset sintético:** 1.000 prompts (500 DOM-PI + 500 docentesDC) gerados por self-instruct; o professor
  respondeu **com RAG** sobre o índice DOM-PI (braço B) e expôs os **top-50 logprobs por token** (cache offline,
  estilo Gemma 2 — subconjunto amostrado do vocabulário).
- **Perda:** `α·CE + (1−α)·T²·KL` (α=0,5, T=2,0). A KL usa a softmax do professor **renormalizada sobre os top-50**.
- Full fine-tune, 3 épocas, lr 1e-5 (cosine), grad-accum 16.

## Resultado (benchmark held-out 100Q, 50 DOM-PI + 50 docentesDC, sem RAG na inferência)
| | ROUGE-L | key-recall |
|---|---|---|
| base 1.5B | 0,185 | 0,366 |
| **este modelo** | **0,363** | **0,717** |

**≈ +96%** em ambas as métricas sobre a base — a destilação internalizou o conhecimento factual do professor nos
pesos do aluno. Aterrar o professor com RAG (braço B) transferiu mais fatos corretos que a versão "zerada" (braço A).

## Uso pretendido e limitações
- **Uso:** assistente factual conciso sobre atos administrativos municipais do Piauí (DOM-PI) e conteúdo docentesDC.
- **Limitações:** conhecimento é um *snapshot* do professor+corpus; para fatos voláteis/atualizáveis, prefira RAG
  (ver Q5). ROUGE-L absoluto modesto (reformulação de frases); o ganho está no conteúdo factual (key-recall).
  Pode alucinar fora do domínio. Não auditado para uso em produção.

## Reprodução
Scripts em `trabalho-final/Q4-destilacao/scripts/` (gerar_dataset_destilacao.py → destilar.py → avaliar_destilacao.py).
