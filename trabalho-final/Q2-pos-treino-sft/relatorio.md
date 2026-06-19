# Relatório — Questão 2: Pós-treino SFT (docentesDC)

## Status: 📋 Planejada (a executar)

## 1. Enunciado
Gerar **≥1.000 pares** instruction/input/output a partir do dataset *docentesDC*. Usar para **Supervised Fine-Tuning
(SFT)**. Avaliar o LLM **antes e depois**.

## 2. Dataset
`vickminari/docentesDC` (HuggingFace): 13.762 registros, split único `train`, campos `text` + `nome_professor`.
~3.300 chars/registro. Domínio: Ciência da Computação (UFPI) — slides, código, artigos, notas. **Não tem pares
instruction/output nativos** → precisam ser gerados.

## 3. Estratégia de geração dos pares (≥1.000)
Duas fases: (a) extrair trechos coerentes de `text`; (b) gerar pares instruction/input/output (ex.: via LLM professor
ou templates) cobrindo definição, explicação de código, Q&A conceitual. Controle de qualidade: deduplicação e
filtragem por tamanho/idioma.

## 4. SFT
- Base: `Qwen2.5-1.5B`. Formato instruction com **máscara de perda só na resposta** (labels=-100 no prompt).
- **Atenção (pegadinha):** evitar o bug de duplo deslocamento de rótulos identificado na Q1 — `input_ids` e `labels`
  alinhados; sanidade: loss inicial ~2-3, não ~11. Opção: TRL `SFTTrainer`.

## 5. Benchmark e métricas
Benchmark próprio da Q2 (a construir, focado em CC/UFPI). Métricas: PPL/CE antes×depois, token-accuracy, e avaliação
qualitativa de instruções respondidas. Comparar com a Q3 (LoRA/QLoRA) sobre o mesmo conjunto de pares.

## 6. Scripts previstos
`gerar_pares_docentes.py`, `sft_docentes.py` (+ sbatch). Infra reutilizável: `avaliar_modelo.py`,
`comparar_resultados.py`, `inferencia_multi.py`.

Detalhes/handoff: ver memória do projeto e §2 de `../relatorio_tecnico_completo.html`.
