# Resultados Q2/Q3 — índice

## Métricas (JSON)
- `heldout_<metodo>_<tam>.json` — avaliação intrínseca no held-out 20% (PPL/CE/token-acc)
- `bench_<metodo>_<tam>.json`   — geração no benchmark CC/UFPI (EM/contains/F1 + geração completa por item)
- `juiz_<metodo>_<tam>.json`    — nota do juiz LLM 1-5 (média, distribuição, por tipo)
- `<metodo>` ∈ {baseline, full, lora, qlora} · `<tam>` ∈ {1.5b, 0.5b}

## Consolidado
- `resumo_q2q3.md` — tabelas antes×depois, custo e arco Q1→Q3 (gerado por `scripts/consolidar_q2q3.py`)

## Figuras (geradas por `scripts/graficos_q2q3.py`)
- `fig_dataset.png`          — distribuição dos 1.500 pares + funil de qualidade do juiz
- `fig_ppl.png`              — PPL no held-out antes×depois
- `fig_terseness.png`        — comprimento médio da resposta (achado central)
- `fig_juiz_tipo.png`        — nota do juiz por tipo de questão
- `fig_custo_qualidade.png`  — custo (VRAM/params) × qualidade (juiz)
- `fig_juiz.png`             — nota do juiz por método e tamanho
