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
- `scripts/pretreino_fullft.py` — Full FT + AdamW 8-bit + early stopping
- `scripts/avaliar_modelo.py`, `comparar_resultados.py`, `inferencia_multi.py` — avaliação
- `scripts/*.sbatch` — jobs SLURM (incluindo os experimentos internos rotulados "q2"/"q3", que são variações de método DENTRO da Q1)

## Modelos
`Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` — variantes: Full FT, LoRA, QLoRA.

## Resultado-chave
Nenhum método de DAPT superou o baseline em perplexidade no held-out (corpus ~50 M tokens é ~200× menor que
o mínimo da literatura). Melhor resultado: **QLoRA 1.5B** (+4,4% PPL, sem colapso). LoRA colapsou ("intruder
dimensions"); Full FT no corpus Teresina aprendeu padrões estruturais (portarias/tabelas) sem capturar isso na PPL.

## Relatório
Seção "Questão 1" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
