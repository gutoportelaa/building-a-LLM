# Relatório — Questão 4: Destilação de Conhecimento

## Status: 📋 Planejada (a executar)

## 1. Enunciado
Investigar modelos usados em destilação. Definir modelos **professor** e **aluno**. Usar um **dataset sintético**.
Criar um benchmark de **100 perguntas**. Analisar a transferência de conhecimento.

## 2. Modelos
- **Professor:** `Qwen2.5-7B-Instruct` (mesma família do aluno → vocabulário/tokenizador alinhados, requisito para
  destilação por logits).
- **Aluno:** `Qwen2.5-1.5B` (ou 0.5B para maior contraste) — mesma base da Q1.

## 3. Dataset sintético
~1.000 prompts (500 DOM-PI da Q1 + 500 docentesDC da Q2). O professor gera **resposta (hard label)** e, quando viável,
**top-k logits (soft labels)**.

## 4. Métodos de destilação
- (a) **Imitação pura (CE)** sobre as respostas do professor;
- (b) **KL-divergência** nos logits (soft targets, temperatura T);
- (c) **Combinada** α·CE + (1−α)·KL (α=0,5).

## 5. Benchmark e métricas
**100 perguntas** (50 DOM-PI + 50 docentesDC). Métricas: PPL do aluno antes/depois, **ROUGE-L vs professor**, e casos
em que o aluno destilado supera o aluno SFT simples. Analisar quanto do conhecimento do professor transfere por
método.

## 6. Observação de viabilidade
A destilação por logits exige o professor 7B em memória ou logits pré-computados; trabalha bem com **modelos mais
controlados** (mesma família, geração offline). Scripts previstos: `gerar_dataset_destilacao.py`, `destilar.py`.

Detalhes: §4 de `../relatorio_tecnico_completo.html`.
