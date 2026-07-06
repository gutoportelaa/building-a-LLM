# Relatório — Questão 3: Pós-treino LoRA/QLoRA (docentesDC)

## Status: ✅ Concluída (mesma cadeia da Q2, métodos LoRA e QLoRA)

## 1. Enunciado
Repetir o pós-treino da Q2 com **LoRA/QLoRA** sobre o *docentesDC*, avaliar antes e depois
e **comparar** com o SFT full (Q2). Se possível, **mais de um modelo** de tamanhos diferentes.

## 2. Estratégia (comparação controlada)
**Mesmos 1.200 pares de treino**, **mesmo benchmark CC/UFPI (30 Q)**, **mesmo loop** da Q2.
A única variável é o método: `sft_docentes.py --method lora|qlora`.
- **LoRA** (r=16, α=32, módulos q/k/v/o/gate/up/down, base bf16) e **QLoRA** (idem + base 4-bit NF4).
- lr 2e-4, 3 épocas, máscara só na resposta. Modelos `Qwen2.5-1.5B-Instruct` e `0.5B-Instruct`.

## 3. Comparação Q2 (full) × Q3 (LoRA/QLoRA) — Qwen2.5-1.5B

| Método | Params treináveis | % | VRAM pico | Tempo | PPL ↓ | F1 | Juiz ↑ |
|---|---|---|---|---|---|---|---|
| **SFT full (Q2)** | 1,54 B | 100% | 15,5 GB | 9 min | **3,56** | **0,283** | 3,37 |
| **LoRA (Q3)** | 18,5 M | **1,18%** | **6,6 GB** | 6 min | 3,65 | 0,282 | 3,40 |
| **QLoRA (Q3)** | 18,5 M | 2,04% | **3,3 GB** | 16 min | 4,32 | 0,240 | **4,07** |

**Leituras:**
1. **LoRA ≈ full FT** em qualidade (PPL 3,65 vs 3,56; F1 0,282 vs 0,283; juiz 3,40 vs 3,37)
   treinando **1,18% dos parâmetros** e usando **43% da VRAM**. É o resultado-livro do PEFT:
   quase-paridade a uma fração do custo.
2. **QLoRA** tem a **menor VRAM (3,3 GB)** — cabe folgado num L4 — e, surpreendentemente,
   o **maior juiz (4,07)**. O motivo está no §4.
3. QLoRA tem a **pior PPL/F1**: a quantização 4-bit + a fusão do adapter em 4-bit
   (rounding) deixam o modelo **menos ajustado à distribuição-alvo**.

> A % treinável do QLoRA aparece "maior" (2,04%) porque o denominador (parâmetros do modelo
> base) encolhe na quantização 4-bit; em valor absoluto é o **mesmo adapter de 18,5 M** do LoRA.

## 4. Por que o QLoRA é o melhor no juiz? (a conservadoria vira vantagem)
Na Q2 mostramos que o SFT **comprime a resposta** (1.5B: 849 → ~150 chars no full/LoRA),
e que essa brevidade **derruba o raciocínio em código**. O QLoRA, por adaptar de forma mais
**conservadora** (base quantizada, gradiente mais ruidoso), **quase não comprimiu** —
comprimento médio **777 chars**, perto do baseline:

| Método | Compr. médio resposta | Juiz código | Juiz total |
|---|---|---|---|
| full FT | 157 chars | 2,5 | 3,37 |
| LoRA | 136 chars | 2,7 | 3,40 |
| **QLoRA** | **777 chars** | **4,5** | **4,07** |

Exemplo (cod01, resposta certa **'I'**):
> **full FT:** *"O"* ❌  · **QLoRA:** *"…o elemento no topo será 'I'."* ✅

Ou seja: a **mesma conservadoria** do PEFT preserva a capacidade de raciocínio do backbone,
evitando o sobre-ajuste ao estilo terso do corpus. O que custa em PPL **ganha** em qualidade percebida.

![Nota do juiz 1-5 por método e tamanho](fig_juiz.png)

## 5. O arco Q1 → Q3 (mesma maquinaria LoRA, veredito oposto)

| Objetivo | LoRA / QLoRA | full-FT |
|---|---|---|
| **CPT bruto (Q1, DOM-PI/Teresina)** | **colapsou** — conservador demais p/ aprender domínio cru | **venceu** (ganho real −12,9%) |
| **SFT instruction (Q3, docentesDC)** | **funciona**; QLoRA dá o **melhor juiz** ao não sobre-ajustar | referência (Q2), melhor PPL |

A **conservadoria** do LoRA/QLoRA é **defeito na Q1** (pré-treino continuado bruto exige
adaptação forte da distribuição → o adapter de baixo rank não dá conta e colapsa) e **virtude
na Q3** (no SFT, adaptar demais ao estilo curto **piora** o raciocínio → adaptar menos
preserva qualidade). Mesmo mecanismo, objetivos diferentes, vereditos opostos — ancorável em
*"LoRA vs Full Fine-tuning: An Illusion of Equivalence"* (intruder dimensions, arXiv:2410.21228).

## 6. Eixo de tamanho (0.5B)
Mesmo padrão, custos ainda menores: full 100%/5,0 GB, LoRA 1,75%/3,0 GB, QLoRA 2,72%/**2,15 GB**.
QLoRA novamente é o mais barato e o que melhor preserva o raciocínio do pequeno backbone.

## 7. Conclusão
Para **instruction tuning** do docentesDC, **LoRA entrega a qualidade do full FT a ~1% do
custo de parâmetros e metade da VRAM**, e **QLoRA é o mais econômico (≈3 GB)** com a **melhor
qualidade percebida** — porque sua adaptação conservadora evita o sobre-ajuste à brevidade que
penaliza o full FT. Combinado com a Q1, fecha-se a lição: *PEFT é conservador — isso colapsa no
pré-treino continuado bruto, mas é exatamente o que se quer no SFT.*

## 8. Reprodução
Mesma cadeia/serviços da Q2: [`../Q2-pos-treino-sft/scripts/README_RUNBOOK.md`].
Métricas em `../Q2-pos-treino-sft/resultados/*.json`; consolidado em `resultados/resumo_q2q3.md`.
