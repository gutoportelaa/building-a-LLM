# Q1 — Pré-treino Continuado (DAPT)

## Enunciado
Fazer pré-treinamento continuado de um LLM considerando o dataset unificado *diariosPrefeituras* (DOM-PI)
e avaliar a qualidade antes e depois. Criar um benchmark com **≥25 perguntas e respostas de referência**.
Métricas exigidas: **perplexidade, entropia cruzada (CE) e acurácia de previsão de tokens**.

## Status: ✅ Concluída

## O que há nesta pasta
- `dompi_qa.jsonl` — benchmark de **49 perguntas** (supera o mínimo de 25)
- `scripts/pretreino_continuado.py` — Full FT padrão
- `scripts/pretreino_lora.py` — LoRA / QLoRA
- `scripts/pretreino_fullft.py` — Full FT + AdamW 8-bit + early stopping (objetivo de treino corrigido)
- `scripts/run_q1_fullft_teresina_v3.sbatch` — job do modelo final (v3)
- `scripts/avaliar_modelo.py`, `comparar_resultados.py`, `inferencia_multi.py` — avaliação
- `scripts/*.sbatch` — jobs SLURM (incluindo os experimentos internos rotulados "q2"/"q3", que são variações de método DENTRO da Q1)

## Modelos — duas respostas (por desenho), ambas positivas
`Qwen/Qwen2.5-1.5B` (e 0.5B nos experimentos iniciais) — variantes Full FT, LoRA, QLoRA. O enunciado pede o
pré-treino com o **dataset completo**; com o objetivo de treino corrigido, ambas as respostas superam o baseline:

1. **Resposta canônica (dataset completo):** Full FT no corpus DOM-PI unificado (v2, objetivo corrigido) —
   held-out PPL 10,03 → **8,90 (−11,3%)**. É a resposta literal ao enunciado e o melhor resultado no held-out geral.
   HF (privado): `gutoportelaa/qwen2.5-1.5b-dompi-fullft-unificado`.
2. **Resposta alternativa (corpus curado):** Full FT no subcorpus de **Teresina** (v3) — held-out 6,91 → **6,02
   (−12,9%)**. HF (privado): `gutoportelaa/qwen2.5-1.5b-dompi-teresina-v3`.

## Versões e métricas (held-out) — bug → corrigido
| Versão | Corpus | Objetivo | held-out PPL | Δ |
|---|---|---|---|---|
| **Full FT unificado v2** (491) | completo | **corrigido** | **8,90** | **−11,3%** ✓ |
| **Full FT Teresina v3** (489) | Teresina | **corrigido** | **6,02** | **−12,9%** ✓ |
| QLoRA unificado v2 (492) | completo | **corrigido** | 9,88 | −1,4% |
| Full FT unificado (487) | completo | bug | 22,22 | +121% |
| QLoRA (421) | completo | bug | 10,47 | +4,4% |
| Full FT Teresina v1 / v2 | Teresina | bug | 8,76 / 12,64 | +27% / +83% |
| LoRA (396/426) | completo | bug | ~1.000 | colapso |

## Resultado-chave
Com o objetivo corrigido, o pré-treino **melhora o modelo nos dois corpora**, e o **Full FT supera o PEFT**:
dataset completo Full FT −11,3% vs QLoRA −1,4%; corpus curado Teresina −12,9%.

**A reviravolta:** as rodadas "degradavam" por um **bug de duplo deslocamento de rótulos** nos três scripts de
treino (`pretreino_fullft.py`, `pretreino_continuado.py`, `pretreino_lora.py`) — o dataset pré-deslocava os `labels`
e o modelo HuggingFace também desloca internamente (objetivo "prever 2 tokens à frente"; loss ~11, PPL interna
~47.000). A tese do "corpus insuficiente" era artefato disso. O bug não só piorava os números — **invertia a
conclusão** (fazia o QLoRA parecer melhor que o full FT). As versões 487/421/v1/v2/LoRA ficam registradas para
evidenciar o impacto do bug.

## Relatório
Seção "Questão 1" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
