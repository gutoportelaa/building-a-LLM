# Resumo consolidado — Q2 (SFT full) × Q3 (LoRA/QLoRA)


## Qwen2.5-1.5b

### Antes × depois (qualidade)

| Modelo | PPL held-out ↓ | EM bench | contains | F1 médio | Juiz 1-5 ↑ |
|---|---|---|---|---|---|
| **Base 1.5b (antes)** | 4.33 | 0.000 | 0.000 | 0.212 | 3.43 |
| SFT full (Q2) 1.5b | 3.56 | 0.000 | 0.000 | 0.283 | 3.37 |
| LoRA (Q3) 1.5b | 3.65 | 0.000 | 0.000 | 0.282 | 3.40 |
| QLoRA (Q3) 1.5b | 4.32 | 0.000 | 0.000 | 0.240 | 4.07 |


### Custo de treino (Q2 full × Q3 PEFT)

| Método | Params treináveis | % treinável | VRAM pico (GB) | Tempo (min) | loss ini→fim |
|---|---|---|---|---|---|
| SFT full (Q2) | 1,543,714,304 | 100.00% | 15.46 | 9.2 | 2.03→1.27 |
| LoRA (Q3) | 18,464,768 | 1.18% | 6.60 | 6.1 | 2.03→0.99 |
| QLoRA (Q3) | 18,464,768 | 2.04% | 3.34 | 16.3 | 2.05→1.01 |

## Qwen2.5-0.5b

### Antes × depois (qualidade)

| Modelo | PPL held-out ↓ | EM bench | contains | F1 médio | Juiz 1-5 ↑ |
|---|---|---|---|---|---|
| **Base 0.5b (antes)** | 5.29 | 0.000 | 0.000 | 0.214 | 2.77 |
| SFT full (Q2) 0.5b | 4.56 | 0.000 | 0.000 | 0.241 | 1.87 |
| LoRA (Q3) 0.5b | 4.64 | 0.000 | 0.000 | 0.248 | 2.13 |
| QLoRA (Q3) 0.5b | 5.72 | 0.000 | 0.000 | 0.195 | 2.53 |


### Custo de treino (Q2 full × Q3 PEFT)

| Método | Params treináveis | % treinável | VRAM pico (GB) | Tempo (min) | loss ini→fim |
|---|---|---|---|---|---|
| SFT full (Q2) | 494,032,768 | 100.00% | 5.00 | 5.2 | 2.42→1.53 |
| LoRA (Q3) | 8,798,208 | 1.75% | 3.00 | 5.1 | 2.42→1.23 |
| QLoRA (Q3) | 8,798,208 | 2.72% | 2.15 | 13.1 | 2.43→1.25 |


## Arco Q1 → Q3 (mesma maquinaria LoRA, objetivos diferentes)

| Objetivo | LoRA | QLoRA | full-FT |
|---|---|---|---|
| **CPT bruto (Q1, DOM-PI/Teresina)** | colapsou (PEFT conservador) | — | venceu (ganho real de domínio) |
| **SFT instruction (Q3, docentesDC)** | ver tabela acima | ver tabela acima | referência (Q2) |
