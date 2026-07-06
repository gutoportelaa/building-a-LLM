# Q2 — Pós-treino SFT

## Enunciado
Gerar **≥1.000 pares** python dicts de perguntas e respostas com instruction/input/output a partir do dataset *docentesDC*. Usar esses pares para pós-treino
**Supervised Fine-Tuning (SFT)**. Avaliar o LLM antes e depois. Considerar mais de um modelo LLM

## Status: ⏳ Pendente

## Estratégia (resumo)
1. **Preparar corpus:** baixar `vickminari/docentesDC` (13.762 registros), filtrar `len>200`, chunking 800–1200 chars → `chunks.jsonl`
2. **Gerar pares (síntese):** LLM (Claude Haiku / GPT-4o-mini) gera 1 par {instruction, input, output} por chunk; distribuição: explicação 30% · factual 25% · código 20% · resumo 15% · comparação 10%
3. **SFT:** `Qwen2.5-1.5B-Instruct`, formato Alpaca/ChatML, loss só nos tokens de output, lr=2e-4, 2–3 épocas, batch efetivo 16
4. **Avaliar:** PPL antes/depois (held-out 20%), ROUGE-L vs referência, benchmark de 30 perguntas (10 conceitual · 10 código · 10 contextual UFPI)

Scripts a criar: `scripts/preparar_docentes.py`, `scripts/gerar_pares_sft.py`, `scripts/sft_docentes.py`.

Detalhamento completo: seção "Questão 2" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
