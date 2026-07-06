# Pós-treino de LLMs no domínio docentesDC — SFT (Q2) e LoRA/QLoRA (Q3)

*Apresentação do desafio · DOM-PI / UFPI-DC · Qwen2.5-1.5B & 0.5B*

---

## Sumário executivo (TL;DR)

A partir de um corpus de material didático de Ciência da Computação (UFPI), geramos
**1.500 pares instrução→resposta** com um professor LLM, fizemos **SFT** (Q2) e repetimos
com **LoRA/QLoRA** (Q3), avaliando **antes e depois** em dois modelos.

- **O SFT adapta ao domínio:** perplexidade no held-out caiu **−18%** (1.5B) e a qualidade
  conceitual subiu (juiz **3,6→4,1**).
- **Achado central:** o SFT **encurta a resposta** (849→~150 chars) — ótimo para definições,
  **ruim para código** (raciocínio some). **PPL baixa ≠ melhor qualidade.**
- **LoRA ≈ full FT** treinando **1,18% dos parâmetros** e usando **43% da VRAM**.
- **QLoRA é o mais barato (3,3 GB) e o melhor no juiz (4,07)** — porque adapta de forma
  conservadora e **não comprime** a resposta.
- **Arco Q1→Q3:** a mesma maquinaria LoRA que **colapsou** no pré-treino contínuo bruto (Q1)
  **funciona** no SFT (Q3) — a conservadoria do PEFT é defeito num caso e virtude no outro.

---

## 1. O desafio

- **Q2 — SFT:** gerar **≥1.000 pares** instruction/input/output do dataset *docentesDC*, fazer
  *Supervised Fine-Tuning* e avaliar o LLM **antes e depois**; considerar **mais de um modelo**.
- **Q3 — LoRA/QLoRA:** repetir o pós-treino com **LoRA/QLoRA** e **comparar** com o SFT full;
  se possível, modelos de **tamanhos diferentes**.

A estratégia foi tratar Q2 e Q3 como **um único experimento controlado**: mesmos pares, mesmo
benchmark, mesmo loop de treino — a única variável é o **método** (`full` × `lora` × `qlora`).

---

## 2. O dataset — `vickminari/docentesDC`

**13.762 registros**, campos `text` + `nome_professor`, **19 professores** de Ciência da
Computação da UFPI. Conteúdo **heterogêneo e ruidoso**, verificado na inspeção:
- **slides** de Estruturas de Dados (pilhas, filas) com **artefatos de OCR** (`Ø`, `§`, palavras
  coladas como `DC/CCN/UFPIErico`);
- **código C** real (pilha dinâmica com `push`/`pop`);
- datasets-exemplo de disciplinas.

**Problema:** o docentesDC **não tem pares instrução/resposta** — só texto corrido. Eles
precisam ser **gerados**.

---

## 3. Processo 1 — Geração dos pares (*grounded self-instruct*)

> **Decisão ① — Grounded self-instruct, não self-instruct puro.**
> No Alpaca, o modelo inventa instruções "do nada" (alucinação alta). Aqui, **cada par nasce de
> um chunk real** do corpus: o professor lê a passagem e produz a pergunta/resposta ancorada
> nela. Reduz alucinação e mantém o conteúdo no domínio da disciplina.

> **Decisão ② — Professor = Qwen2.5-14B no cluster (não API externa).**
> Optou-se pelo 14B local (mesma infra da Q4): offline, sem custo por token, reprodutível e
> coerente com o restante do trabalho.

> **Decisão ③ — Servir o 14B em AWQ numa única GPU.**
> O nó de 2 GPUs estava ocupado por outro job. Em vez de esperar ~15 h na fila, servimos o
> **Qwen2.5-14B-Instruct-AWQ** (quantizado, ~10 GB) com **vLLM em 1 L4** — geração rápida e imediata.

**Pipeline:** `text` (filtro `len>200`) → chunks de 800–1.200 chars → o professor gera um JSON
`{instruction, input, output}` por chunk, com distribuição-alvo de **tipos** (explicação,
factual, código, resumo, comparação); chunks com cara de código são roteados para "código".

> **Decisão ④ — Controle de qualidade em 3 camadas.**
> (1) validação estrutural (JSON válido, campos não-vazios, anti-eco da passagem);
> (2) deduplicação de instruções; (3) **juiz LLM** (o próprio 14B) pontua 1–5 e **descarta < 3**.
> Sobre-geramos ~1,6× para absorver descartes.

**Resultado da geração:** de ~2.400 candidatos, descartamos 247 respostas curtas, 74 duplicatas,
9 JSON inválidos e **440 reprovados pelo juiz** → **1.500 pares limpos**, split **1.200 treino /
300 held-out**.

![Distribuição dos pares e funil de qualidade do juiz](fig_dataset.png)

---

## 4. Processo 2 — Supervised Fine-Tuning

> **Decisão ⑤ — Um único script `sft_docentes.py --method full|lora|qlora`.**
> Garante **comparação controlada** entre Q2 e Q3: mesmo loop, mesmos dados, mesma máscara —
> só muda o método. Sem isso, diferenças de engenharia contaminariam a comparação.

> **Decisão ⑥ — Máscara de perda só na resposta (e sem duplo-shift).**
> `labels = [-100]*prompt + answer`, com `input_ids`/`labels` **alinhados** (o modelo HF faz o
> shift interno). Isso evita o **bug de duplo deslocamento** diagnosticado na Q1. Sanidade
> confirmada: **loss inicial ~2,0** (não ~11).

> **Decisão ⑦ — Dois modelos: Qwen2.5-1.5B e 0.5B (Instruct).**
> Atende "mais de um modelo" e dá um **eixo de tamanho** (curva tamanho×ganho). Usar a variante
> *Instruct* como base é mais honesto/forte do que partir de um base cru.

Formato **ChatML** nativo do Qwen; 3 épocas; lr 1e-5 (full) / 2e-4 (PEFT); batch efetivo 16; bf16.

---

## 5. Processo 3 — Avaliação antes × depois

> **Decisão ⑧ — Três camadas de métrica, no formato em que o modelo treinou.**
> (1) **Intrínseca** no held-out 20%: PPL/CE/token-acc da resposta-alvo (sinal mais limpo de
> adaptação ao domínio). (2) **Geração** no benchmark próprio: EM/contains/F1 + a **geração
> completa** salva (painéis para a banca). (3) **Juiz LLM 1–5** (MT-Bench): sinal semântico para
> instruções abertas, onde EM/F1 punem paráfrases corretas.

> **Decisão ⑨ — Benchmark próprio CC/UFPI, não um benchmark genérico (p.ex. ENEM).**
> 30 questões (10 conceitual / 10 código / 10 contextual UFPI), ancoradas na linguagem real dos
> slides. Mais específico ao domínio do que provas gerais.

> **Decisão ⑩ — Descartar EM/contains do destaque.**
> Deram **0 em todos os braços** (as referências são frases completas; a geração parafraseia) →
> métricas não-informativas aqui. O destaque é **PPL + F1 + juiz**.

---

## 6. Resultados — Q2 (SFT full)

### 6.1 Adaptação ao domínio (held-out)

![PPL no held-out antes×depois](fig_ppl.png)

| Qwen2.5-1.5B | PPL held-out ↓ | F1 (bench) | Juiz 1-5 ↑ |
|---|---|---|---|
| Base (antes) | 4,33 | 0,212 | 3,43 |
| **SFT full** | **3,56** (−18%) | **0,283** (+33%) | 3,37 |

A PPL caiu de forma consistente → **o SFT de fato adaptou o modelo ao domínio**. O juiz, porém,
ficou estável: o baseline *Instruct* já é forte em CC genérica.

### 6.2 Achado central — o SFT comprime a resposta

![Comprimento médio da resposta por método](fig_terseness.png)

O comprimento médio caiu de **849 → 157 chars** (full): o modelo aprendeu o **estilo conciso**
das respostas curtas do docentesDC. Isso cria um **trade-off por tipo de questão**:

![Nota do juiz por tipo de questão](fig_juiz_tipo.png)

| Tipo | Base | SFT full |
|---|---|---|
| conceitual | 3,6 | **4,1** ↑ |
| contextual | 3,3 | 3,5 ↑ |
| código | 3,4 | **2,5** ↓ |

Em **definições**, ser conciso e correto **ajuda**. Em **código**, a brevidade extrema **remove o
raciocínio** e erra. Exemplo verbatim (cod01 — resposta certa: **'I'**):

> **Pergunta:** Considerando uma pilha vazia, qual o topo após `push('O'); push('C'); push('I')`?
>
> **Base 1.5B:** *"Para resolver, vamos analisar passo a passo: `push('O')`→['O']; `push('C')`→['O','C']; `push('I')`→['O','C','I']. O topo é 'I'."* ✅
>
> **SFT full 1.5B:** *"O"* ❌ — terso demais, perdeu o raciocínio e errou.

**Lição:** PPL menor (full ajusta melhor a distribuição-alvo) **não** garante melhor qualidade
em tarefas de raciocínio.

### 6.3 Eixo de tamanho (0.5B)
O 0.5B replica o padrão de forma mais severa: PPL 5,29→4,56, mas o juiz **cai** 2,77→1,87 — o
modelo pequeno, ao encurtar, perde mais raciocínio. A capacidade do backbone modula o efeito.

---

## 7. Resultados — Q3 (LoRA × QLoRA × full)

### 7.1 Custo × qualidade

![Custo (VRAM/params) × qualidade (juiz)](fig_custo_qualidade.png)

| Qwen2.5-1.5B | Params treináveis | % | VRAM pico | PPL ↓ | Juiz ↑ |
|---|---|---|---|---|---|
| SFT full (Q2) | 1,54 B | 100% | 15,5 GB | **3,56** | 3,37 |
| **LoRA (Q3)** | 18,5 M | **1,18%** | **6,6 GB** | 3,65 | 3,40 |
| **QLoRA (Q3)** | 18,5 M | 2,04% | **3,3 GB** | 4,32 | **4,07** |

1. **LoRA ≈ full FT** em qualidade, a **1,18% dos parâmetros** e **43% da VRAM** — o resultado-livro do PEFT.
2. **QLoRA** tem a **menor VRAM (3,3 GB)** e o **maior juiz (4,07)**.
3. QLoRA tem a **pior PPL** (quantização 4-bit + arredondamento na fusão do adapter).

### 7.2 Por que o QLoRA vence no juiz? A conservadoria vira vantagem

![Nota do juiz por método e tamanho](fig_juiz.png)

O QLoRA adapta de forma **mais conservadora** (base quantizada, gradiente mais ruidoso) e por
isso **quase não comprimiu** a resposta (**777 chars**, perto do baseline). Manteve o raciocínio:

| Método | Compr. resposta | Juiz código | Juiz total |
|---|---|---|---|
| full FT | 157 chars | 2,5 | 3,37 |
| LoRA | 136 chars | 2,7 | 3,40 |
| **QLoRA** | **777 chars** | **4,5** | **4,07** |

O que o QLoRA perde em PPL, **ganha** em qualidade percebida — exatamente por **não** sobre-ajustar
o estilo terso do corpus.

---

## 8. O arco Q1 → Q3 (a mesma maquinaria, veredito oposto)

| Objetivo | LoRA / QLoRA | full-FT |
|---|---|---|
| **CPT bruto (Q1, DOM-PI/Teresina)** | **colapsou** — conservador demais p/ aprender domínio cru | **venceu** (ganho real −12,9%) |
| **SFT instruction (Q3, docentesDC)** | **funciona**; QLoRA dá o **melhor juiz** | referência (Q2), melhor PPL |

A conservadoria do PEFT é **defeito na Q1** (o pré-treino contínuo bruto exige adaptação forte da
distribuição — o adapter de baixo rank não dá conta e colapsa) e **virtude na Q3** (no SFT, adaptar
*demais* ao estilo curto **piora** o raciocínio — adaptar *menos* preserva qualidade). Mesmo
mecanismo, objetivos diferentes, vereditos opostos. Ancorável em *"LoRA vs Full Fine-tuning: An
Illusion of Equivalence"* (intruder dimensions, arXiv:2410.21228).

---

## 9. Conclusões

1. **SFT a partir de pares sintéticos *grounded* adapta o modelo ao domínio** (PPL held-out −18%, conceitual ↑) mesmo com corpus ruidoso de OCR.
2. **Cuidado com o que o SFT otimiza:** ele copiou a **brevidade** dos alvos e isso **custou raciocínio em código** — PPL baixa enganou. Métrica única engana; usamos três.
3. **PEFT é a escolha racional:** **LoRA entrega a qualidade do full FT a ~1% do custo de parâmetros**, e **QLoRA é o mais econômico (3,3 GB) com a melhor qualidade percebida**.
4. **A lição transversal (Q1↔Q3):** *PEFT é conservador — ruim para reescrever a distribuição no pré-treino bruto, ideal para o ajuste fino supervisionado.*

---

## 10. Material para apresentação (disponível localmente)

- **Benchmark:** `benchmark/dc_bench.jsonl` (30 questões).
- **Respostas verbatim de todos os modelos:** [`painel_respostas_q2q3.html`](painel_respostas_q2q3.html)
  — as 30 questões × {base, full, LoRA, QLoRA} × {1.5B, 0.5B}, com nota do juiz e F1.
- **Modelos pós-treinados** (6) em `modelos/sft_<metodo>_<tam>/` — prontos para inferência local.
- **Demonstração ao vivo:** `scripts/inferencia_local.py` roda qualquer modelo na sua máquina:
  ```
  python scripts/inferencia_local.py --model modelos/sft_full_1.5b --bench
  python scripts/inferencia_local.py --model modelos/sft_qlora_1.5b --prompt "O que é uma pilha?"
  ```

## 11. Reprodução (cluster)

Cadeia SLURM (L4): `run_q2_gerar_1gpu` → `run_q2q3_treino_aval` → `run_q2q3_juiz_1gpu`,
depois `consolidar_q2q3.py` + `graficos_q2q3.py` + `painel_respostas.py`. Scripts em
`scripts/`; métricas brutas (com geração completa) em `resultados/*.json`; runbook em
`scripts/README_RUNBOOK.md`. Relatórios por questão:
[Q2](relatorio_q2.html) · [Q3](../Q3-pos-treino-lora-qlora/relatorio_q3.html).
