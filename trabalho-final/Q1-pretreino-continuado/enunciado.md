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

## Modelos
`Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` — variantes: Full FT, LoRA, QLoRA.
**Modelo final da Q1:** Full FT 1.5B sobre o corpus curado de Teresina, com o objetivo de treino corrigido (v3).

## Resultado-chave
Com o objetivo de treino corrigido, o **Full FT 1.5B no corpus Teresina (v3) superou o baseline em todas as métricas**:
held-out PPL 6,91 → **6,02 (−12,9%)**, token-accuracy 58,8% → 60,8%, benchmark PPL 7,45 → 7,24 (−2,8%).

**A reviravolta:** as rodadas anteriores "degradavam" por um **bug de duplo deslocamento de rótulos** — o dataset
pré-deslocava os `labels` e o modelo HuggingFace também desloca internamente, fazendo o treino otimizar "prever 2
tokens à frente" (loss ~11, PPL interna ~47.000). A tese do "corpus insuficiente" era artefato desse bug. Corrigido o
alinhamento (`input_ids` = `labels`), o DAPT passou a funcionar como a teoria prevê. Para o corpus geral ruidoso,
**QLoRA** (+4,4%) segue como a opção mais robusta; LoRA colapsou ("intruder dimensions").

## Relatório
Seção "Questão 1" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
