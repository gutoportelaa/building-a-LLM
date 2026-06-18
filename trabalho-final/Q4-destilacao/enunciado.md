# Q4 — Destilação de Conhecimento

## Enunciado
Investigar modelos usados em destilação. Definir modelos **professor** e **aluno**. Usar um **dataset sintético**.
Criar um benchmark de **100 perguntas**. Analisar a transferência de conhecimento.

## Status: ⏳ Pendente

## Estratégia (resumo)
- **Professor:** `Qwen2.5-7B-Instruct` (mesma família do aluno → vocabulário/tokenizador alinhados)
- **Aluno:** `Qwen2.5-1.5B` (ou 0.5B para maior contraste) — mesmo base da Q1
- **Dataset sintético:** 500 prompts DOM-PI (Q1) + 500 docentesDC (Q2); professor gera resposta (hard label) + top-k logits (soft labels)
- **Métodos:** (a) imitação pura CE; (b) KL-divergência em logits; (c) combinada α·CE+(1−α)·KL, α=0.5
- **Benchmark 100 perguntas:** 50 DOM-PI + 50 docentesDC. Métricas: PPL aluno antes/depois, ROUGE-L vs professor, casos em que o aluno destilado supera o aluno SFT simples

Scripts a criar: `scripts/gerar_dataset_destilacao.py`, `scripts/destilar.py`.

Detalhamento: seção "Questão 4" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
