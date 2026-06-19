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

## Modelos — duas respostas (por desenho)
`Qwen/Qwen2.5-1.5B` (e 0.5B nos experimentos iniciais) — variantes Full FT, LoRA, QLoRA. O enunciado pede o
pré-treino com o **dataset completo**, então apresentamos duas respostas complementares:

1. **Resposta canônica (dataset completo):** Full FT no corpus DOM-PI unificado — apresentada *mesmo degradada*
   (+121% PPL) por ser literalmente o que o enunciado pede e por evidenciar a dificuldade do full FT sobre corpus
   ruidoso. HF (privado): `gutoportelaa/qwen2.5-1.5b-dompi-fullft-unificado`.
2. **Resposta alternativa (solução):** Full FT no subcorpus curado de **Teresina** com o objetivo de treino
   corrigido (**v3**) — o primeiro a superar o baseline. HF (privado): `gutoportelaa/qwen2.5-1.5b-dompi-teresina-v3`.

## Versões e métricas (held-out)
| Versão | Corpus | Épocas | Objetivo | held-out PPL | Δ |
|---|---|---|---|---|---|
| Full FT unificado (canônico) | completo | 1 | bug | 22,22 | +121% |
| QLoRA | completo | 1 | bug | 10,47 | +4,4% |
| Full FT Teresina v1 | Teresina | 1 | bug | 8,76 | +27% |
| Full FT Teresina v2 (+freeze) | Teresina | 3 | bug | 12,64 | +83% |
| **Full FT Teresina v3** | Teresina | 2 | **corrigido** | **6,02** | **−12,9%** ✓ |

## Resultado-chave
Com o objetivo de treino corrigido, o **v3 superou o baseline em todas as métricas**: held-out PPL 6,91 → **6,02
(−12,9%)**, token-accuracy 58,8% → 60,8%, benchmark PPL 7,45 → 7,24.

**A reviravolta:** as rodadas anteriores "degradavam" por um **bug de duplo deslocamento de rótulos** nos três
scripts de treino (`pretreino_fullft.py`, `pretreino_continuado.py`, `pretreino_lora.py`) — o dataset pré-deslocava os
`labels` e o modelo HuggingFace também desloca internamente (objetivo "prever 2 tokens à frente"; loss ~11, PPL
interna ~47.000). A tese do "corpus insuficiente" era artefato disso. Como só o v3 foi treinado com o objetivo
correto, os números das demais versões são *limites inferiores*. Para o corpus completo ruidoso, **QLoRA** segue
como a opção mais robusta entre as bugadas; LoRA colapsou ("intruder dimensions").

## Relatório
Seção "Questão 1" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
