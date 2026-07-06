# Relatório — Questão 2: Pós-treino SFT (docentesDC)

## Status: ✅ Concluída (treino e avaliação executados no cluster L4)

## 1. Enunciado
Gerar **≥1.000 pares** instruction/input/output a partir do dataset *docentesDC*;
usar para **Supervised Fine-Tuning (SFT)**; avaliar o LLM **antes e depois**;
considerar **mais de um modelo**.

## 2. Dataset
`vickminari/docentesDC` (HuggingFace): **13.762 registros**, split único `train`,
campos `text` + `nome_professor`, **19 professores** (Ciência da Computação / UFPI).
Mediana ~3.059 chars/registro. Conteúdo heterogêneo: **slides** de Estruturas de Dados
(pilhas, filas — com artefatos de OCR), **código C** (pilha dinâmica com push/pop) e
datasets-exemplo. **Não tem pares instruction/output nativos** → foram gerados.

## 3. Geração dos pares — *grounded self-instruct* (1.500 pares)
Cada par nasce de um **chunk real** do corpus (reduz alucinação vs. self-instruct puro):
- **Professor:** `Qwen2.5-14B-Instruct-AWQ` servido com **vLLM** em **1 GPU L4** (TP=1) —
  evitou a fila do nó de 2 GPUs.
- **Chunking:** `len(text)>200` → janelas de 800–1.200 chars em quebra natural.
- **Distribuição final:** explicação 559 · código 367 · resumo 225 · comparação 198 ·
  factual 151 (chunks com cara de código roteados para "código").
- **Controle de qualidade (3 camadas):** validação estrutural → dedup de instruções →
  **juiz LLM (1–5)**. Descartes: 247 respostas curtas, 74 duplicadas, 9 JSON inválido,
  **440 reprovados pelo juiz** (score 1–2). Distribuição de score dos aprovados:
  3→599, 4→492, 5→538. Resultado: **1.500 pares limpos** → **1.200 treino / 300 held-out**.

Scripts: `gerar_pares_docentes.py` · `run_q2_gerar_1gpu.sbatch`.

## 4. SFT
- **Modelos:** `Qwen2.5-1.5B-Instruct` e `Qwen2.5-0.5B-Instruct` (eixo de tamanho →
  "mais de um modelo").
- **Formato:** ChatML nativo do Qwen; **máscara de perda só na resposta**
  (`labels=[-100]*prompt + answer`), `input_ids`/`labels` alinhados.
- **Sanidade (pegadinha da Q1 evitada):** loss inicial **2,03** (1.5B) / **2,42** (0.5B) —
  não ~11 → sem o bug de duplo-shift. 3 épocas, lr 1e-5, batch efetivo 16, bf16.
- Script único `sft_docentes.py --method full` (LoRA/QLoRA = Q3, mesmo loop → comparação controlada).

## 5. Resultados — antes × depois (Qwen2.5-1.5B)

| Modelo | PPL held-out ↓ | F1 médio (bench) | Juiz 1-5 ↑ | Compr. médio resposta |
|---|---|---|---|---|
| **Base 1.5B (antes)** | 4,33 | 0,212 | 3,43 | 849 chars |
| **SFT full (Q2)** | **3,56** (−18%) | **0,283** (+33%) | 3,37 | 157 chars |

- **PPL no held-out caiu 18%** — evidência limpa de que o SFT **adaptou o modelo ao
  domínio** docentesDC (a métrica menos enviesada, no conjunto reservado).
- **F1 lexical subiu 33%** no benchmark CC/UFPI.
- O **juiz** ficou praticamente igual (3,43→3,37): o baseline Instruct **já é forte** em CC
  genérica; o ganho de domínio não se traduz em vantagem clara num juiz semântico.
- `EM`/`contains` foram **0 em todos os braços** (referências são frases completas e a
  geração parafraseia) → métricas não-informativas aqui; usamos **PPL + F1 + juiz**.

### Achado central — SFT comprime as respostas (PPL ≠ qualidade)
O comprimento médio da resposta caiu de **849 → 157 chars** após o SFT full: o modelo
aprendeu o **estilo conciso** das respostas curtas do docentesDC. Isso explica o trade-off
medido pelo juiz **por tipo**:

| Tipo | Base | SFT full |
|---|---|---|
| conceitual | 3,6 | **4,1** ↑ |
| contextual | 3,3 | **3,5** ↑ |
| código | 3,4 | **2,5** ↓ |

Em definições conceituais, ser conciso e correto **ajuda**. Em questões de **código**, a
brevidade extrema **remove o raciocínio** e erra. Exemplo (cod01, "topo após push('O'),
push('C'), push('I')"; resposta certa: **'I'**):

> **Base 1.5B:** *"Para resolver, vamos analisar passo a passo: push('O') → ['O']; push('C') → …"* (raciocínio correto)
> **SFT full 1.5B:** *"O"* ❌ (terso demais → perdeu o raciocínio e errou)

Ou seja: o **full FT ajusta melhor a distribuição-alvo (menor PPL)**, mas o próprio
ajuste à brevidade **degrada tarefas que exigem passos**. Lição clássica de SFT:
**PPL baixa não garante melhor qualidade** em tarefas de raciocínio.

![Nota do juiz 1-5 por método e tamanho](fig_juiz.png)

### Eixo de tamanho (0.5B)
O 0.5B replica o padrão de forma mais severa: PPL 5,29→4,56, mas o juiz **cai** 2,77→1,87
(o modelo pequeno, ao encurtar, perde mais raciocínio). Confirma que a capacidade do
backbone modula quanto o SFT ajuda × atrapalha.

## 6. Conclusão
O SFT a partir de 1.500 pares *grounded self-instruct* **adaptou os modelos ao domínio**
docentesDC (PPL held-out −18%, conceitual ↑), com a ressalva honesta de que a **compressão
do estilo** custa desempenho em código e que um baseline Instruct forte limita o ganho
visível no juiz. A comparação com PEFT (LoRA/QLoRA) e o arco com a Q1 estão na
[Questão 3](../Q3-pos-treino-lora-qlora/relatorio.md) — onde a **conservadoria do QLoRA
vira vantagem** justamente por comprimir menos.

## 7. Reprodução
Runbook: [`scripts/README_RUNBOOK.md`]. Cadeia: `run_q2_gerar_1gpu` → `run_q2q3_treino_aval`
→ `run_q2q3_juiz_1gpu` → `consolidar_q2q3.py`. Métricas brutas em `resultados/*.json`
(com a geração completa por item), consolidado em `resultados/resumo_q2q3.md`.
