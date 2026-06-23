---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
language: [pt]
tags: [knowledge-distillation, cross-tokenizer, universal-logit-distillation, qwen2.5, distillation]
pipeline_tag: text-generation
---

# qwen2.5-1.5b-cross-tokenizer-uld

Aluno destilado da **Questão 4 / frente ULD** (Construindo um LLM, UFPI-DC): **destilação logit cross-tokenizer**
(Universal Logit Distillation, Boizard et al., arXiv:2402.12030) de um professor de **outra família**.

## Como foi treinado
- **Professor:** `HuggingFaceH4/zephyr-7b-beta` (arquitetura Mistral, **tokenizer diferente** do Qwen; escolhido por ser
  *ungated*). Aluno: `Qwen/Qwen2.5-1.5B`, braço B (RAG).
- **Método `uldcomb`** = α·CE + (1−α)·ULD. A perda ULD, por posição, **ordena** as distribuições de probabilidade do
  aluno e do professor e minimiza a L1 entre os vetores ordenados — **invariante ao vocabulário** (permite logit KD entre
  famílias sem tokenizador comum). Implementação *research-grade* com alinhamento posicional aproximado.

## Resultado (benchmark de 100Q, DOM-PI + docentesDC, sem RAG na inferência)
| Aluno (cross-família) | método | ROUGE-L | key-recall |
|---|---|---|---|
| black-box (CE puro) | SFT texto | 0,270 | 0,490 |
| **este modelo (uldcomb)** | CE + ULD | **0,330** | **0,504** |

O sinal de distribuição cross-tokenizer **supera o black-box puro**. Ainda fica **abaixo do white-box mesma-família**
(KR 0,717) — destilar dentro da família com logits segue o teto.

## Limitações
Implementação aproximada do ULD (alinhamento por truncagem; o ULD fiel usa transporte ótimo). Modelo de pesquisa, não
auditado para produção.

## Reprodução
`gerar_dataset_crossfamilia.py` (com `--topk`) → `destilar.py --method uldcomb` → `avaliar_destilacao.py`.
Ver `trabalho-final/Q4-destilacao/` (relatório §10/§4.6).
