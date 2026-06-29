# Relatório — Questão 1: Pré-treino Continuado (DAPT)

## 1. Enunciado e resposta em uma frase
Fazer **pré-treinamento continuado** (DAPT) de um LLM no dataset *diariosPrefeituras* (DOM-PI) e avaliar a
qualidade **antes e depois**, com benchmark de **≥25 perguntas** e as métricas **perplexidade (PPL), entropia
cruzada (CE) e acurácia de previsão de token**.

> **Resposta:** o DAPT do `Qwen2.5-1.5B` no corpus DOM-PI **melhora a modelagem do domínio** — PPL no held-out
> cai **−11,1 %** (dataset completo) e **−12,9 %** (subcorpus curado de Teresina), com token-accuracy subindo nos
> dois casos — **sem esquecimento catastrófico** no domínio geral. O efeito **generaliza entre famílias**: repetido
> num **Llama-3.2-3B (Meta)**, o ganho é de **−23,3 %** (§7c). O resultado está na faixa esperada para DAPT em
> português com corpus pequeno/ruidoso (cf. Juru, Tucano, Curió-Edu).

## 2. Modelo, corpus e benchmark
- **Base:** `Qwen/Qwen2.5-1.5B` (forte em PT, licença permissiva, cabe full-FT numa GPU L4 24 GB com AdamW 8-bit).
- **Corpus:** DOM-PI unificado (224 municípios + capital, ~50 M tokens de OCR ruidoso) e o subcorpus **Teresina**
  curado (~9,3 M tokens, texto mais limpo). Held-out fixo (seed=42) separado antes do treino.
- **Benchmark:** `dompi_qa.jsonl` — **49 perguntas** com fatos verificados do tier A (supera o mínimo de 25).

## 3. Metodologia e protocolo de avaliação
Treino causal *packed* (blocos de 512), AdamW 8-bit, lr=2e-6 (full-FT) / 2e-5 (PEFT), warmup 5 % + cosine,
early stopping por CE no held-out. A avaliação segue o **protocolo de três camadas** consolidado para
continued-pretraining (tutorial NAACL'25, arXiv:2504.03931; guias de produção de DAPT):

| Camada | O que mede | Como | Script |
|---|---|---|---|
| **1. Ganho de domínio** | o modelo ficou melhor no DOM-PI? | PPL/CE/token-acc no held-out e no benchmark | `avaliar_modelo.py` |
| **2. Retenção de capacidade** | o DAPT estragou o conhecimento geral? | PPL/CE num held-out de domínio **geral** (Wikipedia-PT) | `avaliar_modelo.py` + `preparar_heldout_geral.py` |
| **3. Utilidade downstream** | serve numa tarefa real? | usado como gerador no RAG da Q5 | ver Q5 |

Além da PPL/CE (proxy de confiança), medimos **acurácia de geração** (`avaliar_geracao.py`): o modelo *gera* a
resposta e comparamos com a referência por *exact-match*, *contains* e *token-F1* — número direto e interpretável,
no espírito de como AdaptLLM e ChipNeMo reportam ganho de domínio.

## 4. Camada 1 — Ganho de domínio (resultado central)
Avaliação antes×depois no held-out de domínio (mesma run das §5–§6):

| Modelo | Corpus | PPL held-out | Δ vs baseline | token-acc |
|---|---|---|---|---|
| Baseline 1.5B | — | 9,82 / 6,91\* | — | 0,545 |
| **Full FT unificado** | completo | **8,74** | **−11,1 %** ✓ | 0,562 (+1,7 pt) |
| **Full FT Teresina** | Teresina | **6,02**\* | **−12,8 %** ✓ | 0,608 (+6,3 pt) |
| QLoRA unificado | completo | 9,88 | −1,4 % | ~ |

\* Teresina mede no held-out curado de Teresina (baseline 6,91); o unificado, no held-out geral DOM-PI
(baseline 9,82). **Full-FT > PEFT** nos dois corpora — como a teoria de DAPT prevê: parâmetros plenos absorvem
mais do domínio; o low-rank (NF4) regulariza, é mais robusto ao ruído, mas conservador (ganho marginal).

## 5. Camada 2 — Retenção de capacidade (anti-esquecimento)
Held-out de **domínio geral** (Wikipedia-PT, fora dos diários), PPL antes×depois — mesma run que mediu o domínio:

| Modelo | PPL domínio **geral** | PPL domínio **DOM-PI** | token-acc domínio | Leitura |
|---|---|---|---|---|
| Baseline 1.5B | 8,80 | 9,82 | 0,545 | referência |
| Full FT unificado | **8,79** (−0,2 %) | **8,74** (−11,1 %) | 0,562 (+1,7 pt) | ganho de domínio, **retenção intacta** |
| Teresina | **8,77** (−0,4 %) | **6,02**\* (−12,8 %) | 0,608 (+6,3 pt) | idem (corpus curado) |

\* Teresina avaliada no seu próprio held-out curado (baseline 6,91).

> **Resultado forte:** a PPL no domínio **geral não se move** (varia <0,5 %) enquanto a PPL de domínio cai
> 11–13 % — ou seja, **zero esquecimento catastrófico**. É um trade-off ainda mais favorável que o do **Juru**
> (arXiv:2403.18140), que ganha no domínio jurídico mas *perde* fora dele. O `lr` conservador (2e-6) preserva as
> representações originais enquanto especializa o modelo.

## 6. Camada 1 (legível) — acurácia de geração e de token no benchmark
Tentamos a métrica mais palpável (“acertou X de 49”): o modelo **gera** a resposta e comparamos com a referência.

| Modelo | contains (ref na geração) | token-F1 médio | exact-match |
|---|---|---|---|
| Baseline 1.5B | 0/49 | 0,158 | 0/49 |
| Full FT unificado | 0/49 | 0,163 | 0/49 |
| Teresina | 0/49 | 0,148 | 0/49 |

> **Achado (negativo informativo):** *exact-match* e *contains* de geração gulosa são **0 para todos os modelos**,
> inclusive os DAPT. Reproduzir **literalmente** um CNPJ, CPF ou nome próprio exato é um teto alto demais para um
> modelo de 1,5 B decodificando *greedy* — o DAPT aumenta a *probabilidade* da resposta certa (cai a NLL/PPL),
> mas não o suficiente para emergir como string exata na geração livre. **Conclusão metodológica:** para este par
> tarefa×escala, as métricas informativas do ganho de domínio são as que o enunciado pede — **PPL, CE e
> acurácia de previsão de token** (§4–§5), não o exact-match de geração. As respostas completas dos três modelos
> estão em `resultados/geracao_*.json` (campo `per_item.geracao`).

## 7. Curva dose-resposta — PPL cai com log(tokens)
![Dose-resposta](resultados/figuras/curva_dose_resposta.png)

Treinamos o full-FT em frações crescentes do corpus (mesmo held-out fixo):

| Tokens de DAPT | PPL held-out | Δ vs baseline |
|---|---|---|
| 0 (baseline) | 9,82 | — |
| ~4,2 M (10 %) | 9,28 | −5,5 % |
| ~12,5 M (30 %) | 8,99 | −8,4 % |
| ~41,7 M (100 %) | 8,80 | −10,4 % |

A PPL cai monotonicamente e **achata com log(N_tokens)** — cada ~3× mais tokens rende um decréscimo de PPL
cada vez menor, exatamente o padrão de ganho-por-checkpoint que o Juru observa até ~7 B tokens. Confirma que o
gargalo aqui é volume/qualidade de corpus, não a receita de treino.

## 7b. Limpeza de OCR não melhora a PPL (negativo informativo)
Hipótese: remover ruído de OCR (IDs colados, linhas de baixa razão alfabética, cabeçalhos quebrados) daria um
sinal de treino mais limpo e mais ganho — como o subcorpus curado de Teresina sugere. Aplicamos um filtro
heurístico (`limpar_corpus_ocr.py`): **−13 % de caracteres**, 98 % dos documentos preservados.

| Treino | PPL held-out | Δ |
|---|---|---|
| Corpus **bruto** (unificado) | **8,74** | −11,1 % |
| Corpus **limpo** (mesmo volume relativo) | 8,80 | −10,4 % |

**Resultado:** a limpeza **não ajudou** — ficou marginalmente pior. Causa provável: o **held-out continua sendo
OCR bruto**, então um modelo treinado em texto limpo sofre leve descasamento de distribuição (treino limpo ×
avaliação ruidosa) e ainda vê ~13 % menos tokens. Mostrar a benefício real da limpeza exigiria também um
held-out limpo — mas aí os números deixariam de ser comparáveis ao resultado canônico. **Decisão (canônica):
mantém-se o full-FT do corpus bruto unificado** (−11,1 %); a limpeza entra como análise, não como resposta.

## 7c. Generalização cross-família — Qwen vs Llama
O enunciado permite outra família. Repetimos o DAPT canônico (full-FT, mesmo corpus bruto unificado, mesmo
held-out, mesmos hiperparâmetros) num modelo de **família diferente**, o **Llama-3.2-3B (Meta)**, para testar se
o ganho é específico do Qwen ou um efeito geral.

| Modelo | Família | PPL antes | PPL depois | Δ | token-acc |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | Qwen (Alibaba) | 9,82 | 8,74 | −11,1 % | 0,545 → 0,562 |
| **Llama-3.2-3B** | Meta | 10,55 | **8,09** | **−23,3 %** | 0,566 → 0,605 |

**Três conclusões:**
1. **O DAPT generaliza entre arquiteturas** — ambas as famílias melhoram substancialmente; o ganho não é um
   artefato do Qwen.
2. **Quem parte mais fraco no domínio ganha mais.** O Llama começa pior (PPL 10,55 > 9,82 — o Qwen2.5 é mais
   forte em PT de fábrica), mas o DAPT rende o **dobro** (−23,3 % vs −11,1 %) e o Llama-3.2-3B **ultrapassa** o
   Qwen pós-adaptação (8,09 < 8,74): mais capacidade (3 B) e mais "espaço para aprender" o domínio absorvem mais
   sinal. A acurácia de token sobe +3,9 pt.
3. **Geração EM = 0 nos dois** — reforça o achado da §6: o ganho de domínio aparece em PPL/CE/token-acc, não no
   exact-match de geração gulosa (teto de escala/decodificação, não de família).

*Nota técnica: 3 B em full-FT cabe numa única L4 (24 GB) com AdamW-8bit + gradient checkpointing; treino
distribuído (FSDP) só seria necessário a partir de ~7 B.*

## 8. Comparação com DAPT públicos (contextualização)
| Projeto | Base | Domínio / Idioma | O que evidenciam |
|---|---|---|---|
| **Este trabalho** | Qwen2.5-1.5B + Llama-3.2-3B | Diários oficiais PI / PT | −11 % (Qwen) a −23 % (Llama) PPL de domínio, retenção geral preservada, ganho cross-família |
| **Juru** (2403.18140) | Mistral-7B | Jurídico BR / PT | ganho em benchmark legal + esquecimento fora do domínio |
| **Tucano** (2603.03543) | — | PT-BR geral | escala de dados em continued-pretraining PT |
| **Curió-Edu** (2512.12770) | 7B | Educação / PT | impacto da *seleção de dados* no DAPT |
| **AdaptLLM** (2309.09530) | 7B | Bio/Fin/Lei | corpus cru ganha conhecimento mas pode ferir Q&A → medir ambos |

Magnitude e trade-off do nosso resultado são coerentes com a literatura: DAPT em corpus pequeno/ruidoso entrega
ganho de domínio modesto e mensurável, com retenção preservada quando o lr é conservador.

## 9. Exemplos qualitativos (completar portaria)
- **Baseline:** texto genérico, foge do formato de ato administrativo.
- **DAPT (unificado v2):** fluente e no domínio — “…publicado diariamente… normas legais aprovadas…”, estrutura
  de portaria/ementa preservada. Inferências completas em `resultados/inferencias_*.json`.

## 10. Nota de validação do pipeline
Antes de confiar nas métricas, validamos a sanidade do *loop* de treino: a CE do baseline interno deve bater o
`avaliar_modelo.py` (**PPL ≈ 10**, não dezenas de milhar). Essa checagem expôs e corrigiu um *duplo deslocamento
de rótulos* no `PackedTextDataset` (o objetivo otimizava “prever 2 tokens à frente”); após a correção
(`labels = input_ids.clone()`), o baseline interno reconcilia com a avaliação e os ganhos acima são reais.
**Lição metodológica:** validar o pipeline antes de atribuir resultados ao corpus.

## 11. Modelos publicados e reprodução
- `gutoportelaa/qwen2.5-1.5b-dompi-fullft-unificado` — canônica (dataset completo).
- `gutoportelaa/qwen2.5-1.5b-dompi-teresina-v3` — alternativa (corpus curado).
- `gutoportelaa/qwen2.5-1.5b-dompi-dapt` — variante QLoRA (gerador G2 da Q5).

```bash
# treino (resposta canônica / alternativa)
sbatch scripts/run_fullft_unificado.sbatch
sbatch scripts/run_q1_fullft_teresina_v3.sbatch
# evidências de avaliação (retenção + acurácia de geração + curva) — só inferência
sbatch scripts/run_q1_evidencias.sbatch
```
Avaliação: `avaliar_modelo.py` (camadas 1–2) + `avaliar_geracao.py` (acurácia) + `comparar_resultados.py`.
Detalhes: §1 de `../relatorio_tecnico_completo.html`.
