# Q3 — Pós-treino LoRA / QLoRA

## Enunciado
Repetir o experimento de pós-treino da Q2 usando **LoRA e/ou QLoRA**. Avaliar antes e depois.
Se possível, considerar **mais de um modelo** com tamanhos diferentes.

## Status: ⏳ Pendente (reaproveita os pares gerados na Q2)

## Distinção em relação à Q1
Na Q1 aplicamos LoRA/QLoRA para **pré-treino continuado** (next-token, corpus bruto) — onde o LoRA colapsou.
Na Q3 aplicamos LoRA/QLoRA para **instruction tuning (SFT)** — loss só nos tokens de output, sinal de supervisão
muito mais estruturado. Por isso espera-se aprendizado estável mesmo com lr=2e-4 (Alpaca-LoRA).

## Estratégia (resumo)
- **LoRA e QLoRA** sobre `Qwen2.5-1.5B-Instruct` (r=16, alpha=32, módulos q/k/v/o/gate/up/down, lr=2e-4)
- **Segundo modelo:** `Qwen2.5-0.5B-Instruct` (comparação por tamanho)
- Adaptar `pretreino_lora.py` (Q1) para o objetivo SFT com máscara de loss
- Comparar Q2 (Full FT) × Q3 (LoRA) × Q3 (QLoRA): parâmetros treináveis, VRAM, tempo, qualidade

Detalhamento: seção "Questão 3" em [`../relatorio_tecnico_completo.html`](../relatorio_tecnico_completo.html).
