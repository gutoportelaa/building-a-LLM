# Relatório — Questão 3: Pós-treino LoRA/QLoRA (docentesDC)

## Status: 📋 Planejada (a executar)

## 1. Enunciado
Repetir o pós-treino da Q2 usando **LoRA/QLoRA** sobre o *docentesDC* e **comparar** com o SFT full da Q2
(full fine-tuning vs PEFT).

## 2. Estratégia
- Reaproveitar **o mesmo conjunto de ≥1.000 pares** gerado na Q2 → comparação controlada (só muda o método).
- Base `Qwen2.5-1.5B`. **LoRA** (rank 16, α 32) e **QLoRA** (NF4 4-bit). lr ~2e-5, warmup 5%.
- Mesma máscara de perda só na resposta; mesmo objetivo corrigido (sem duplo shift — ver Q1).

## 3. Comparação planejada (Q2 vs Q3)
| Dimensão | SFT full (Q2) | LoRA/QLoRA (Q3) |
|---|---|---|
| Parâmetros treinados | todos (~1,5 B) | adaptadores (~0,5–2%) |
| Memória/custo | maior | menor (QLoRA cabe folgado no L4) |
| Risco | esquecimento | "intruder dimensions" se lr alto |
| Métrica esperada | melhor ajuste | quase-paridade com fração do custo |

## 4. Métricas
PPL/CE/token-accuracy antes×depois no benchmark da Q2; custo (memória, tempo) e qualidade qualitativa. Hipótese:
QLoRA atinge boa fração do ganho do SFT full com custo muito menor — análoga à lição da Q1 (PEFT conservador, full-FT
melhor quando viável).

## 5. Scripts previstos
`pretreino_lora.py` (já existe, com objetivo corrigido) adaptado para formato instruction; sbatches LoRA/QLoRA.

Detalhes: §3 de `../relatorio_tecnico_completo.html`.
