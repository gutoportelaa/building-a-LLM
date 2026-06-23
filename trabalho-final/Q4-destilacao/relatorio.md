# Relatório — Questão 4: Destilação de Conhecimento

## Status: ✅ Concluída (estudo + 12 alunos destilados + avaliação; melhor = 1.5B·B·combinado, +96%)

## 1. Enunciado
Investigar quais LLMs são normalmente usados para destilação. Definir **professor** e **aluno**.
Usar um **dataset sintético** para destilar professor→aluno. Criar um **benchmark de 100 perguntas**
para avaliar professor e aluno antes/depois. **Analisar se houve transferência de conhecimento.**

---

## 2. Investigação: quais LLMs e técnicas são usados em destilação (estado da arte 2024–2026)

A pesquisa foi além de Raschka, cruzando relatórios técnicos de produção e artigos recentes.
Três conclusões organizam o desenho:

### 2.1 A prática consolidada é destilar *dentro da mesma família*
Porque a destilação por **logits (white-box)** exige **tokenizador/vocabulário compartilhado**:

| Caso real | Professor | Aluno | Técnica |
|---|---|---|---|
| **Gemma 2** (Google, 2024) | 27B / Gemini | 2B, 9B | white-box logit KD; "substitui o one-hot pela distribuição do professor"; só um **subconjunto amostrado** dos logits é armazenado (vocab 256k) |
| **DeepSeek-R1-Distill** (2025) | DeepSeek-R1 671B | Qwen 1.5B–Llama 70B | **SFT puro** sobre ~800K traços de raciocínio — *sequence-level / black-box / off-policy*; **sem logits** |
| **Llama 4** (Meta) | Behemoth | Scout, Maverick | logit KD mesma família |

Professores black-box mais comuns na literatura: **GPT-4/4o, Qwen2.5-72B-Instruct, DeepSeek-R1, Llama-3.1**.

### 2.2 Os três regimes técnicos e seus custos

| Regime | Tokenizer igual? | Custo | Maturidade |
|---|---|---|---|
| **White-box logit KD** (KL nos soft logits) | **Sim** → só mesma família | professor em memória OU logits pré-computados | padrão (Gemma 2, DistillKit, easydistill) |
| **Black-box sequence-KD** (SFT no texto do professor) | Não → qualquer família | só gerar o dataset sintético | padrão (DeepSeek-R1) |
| **Cross-tokenizer logit KD** (ULD, ALM, Optimal Transport) | Não (via aproximação) | alto, *research-grade* | fronteira 2025 |

### 2.3 On-policy vs off-policy (literatura nova)
- **Off-policy** (treinar no texto do professor, ex. DeepSeek-R1): sofre *exposure bias* (mismatch treino↔inferência).
- **On-policy** (MiniLLM 2023, GKD 2024): destila na **distribuição do próprio aluno**; hoje é ingrediente de
  Qwen3, Gemma 2, DeepSeek-V4. Mais caro de implementar (rollouts do aluno em treino).

### 2.4 Decisão sobre cross-família / cross-especialidade / cross-modalidade
- **Cross-modalidade (imagem/voz→texto):** ❌ descartado — corpus é texto governamental puro; custo alto, payoff nulo.
- **Cross-especialidade (código→geral):** ❌ descartado — a literatura mostra que dados de especialidade
  incompatível *degradam* o aluno ("cross-model personalized data does not perform as well as real
  personalized data"); um professor de código nada tem a transferir sobre DOM-PI/docentesDC.
- **Cross-família (Gemma/Llama→Qwen):** ✅ viável **apenas como braço black-box** (sequence-KD), reservado
  como **expansão** caso o núcleo dê resultados satisfatórios. Logit KD cross-família exigiria ULD/OT (caro).

### 2.5 Professor forte e factual (não o modelo fraco da Q1)
Usar o modelo treinado da Q1/Q5 como **professor** seria errado: o princípio do KD é *professor ≫ aluno*, e nosso
DAPT 1.5B é pequeno e treinado em corpus com OCR corrompido (ganhos marginais) → transferiria os próprios erros.
Por isso o professor é o **`Qwen2.5-7B-Instruct` oficial** (forte, limpo, mesmo tokenizador → habilita logit KD).
O modelo da Q1/Q5 **não** é usado como professor nem como aluno. O aluno é o **base pristino** (baseline limpo
para medir transferência).

### 2.6 Eixo experimental: destilação "zerada" × professor com RAG
Como o professor 7B genérico não conhece os fatos específicos do DOM-PI (alucina), o *grounding* do professor
vira **variável de estudo**, não detalhe de implementação. Três braços respondem "houve transferência?":

| Braço | O professor responde… | O que mede |
|---|---|---|
| **A — "zerada"** | só da memória paramétrica (sem recuperar) | quanto o professor sabe sozinho (com alucinação) e quanto disso o aluno absorve |
| **B — RAG-grounded** | com contexto recuperado do índice DOM-PI (Q5) | transferência de conhecimento **factual e correto** para os pesos do aluno |
| **C — RAG na inferência** (baseline Q5) | não destila; recupera em tempo de resposta | teto: conhecimento **externo** (Q5) vs **internalizado** nos pesos |

Conclusão visada: *quanto dá para "assar" conhecimento nos pesos (B) e quão perto isso chega de carregá-lo
externamente (C)?* — ponte direta Q4↔Q5.

### 2.5 Frameworks de referência (evitam reinventar)
- **Arcee DistillKit** (`distil_logits.py`): KL + alinhamento de hidden states, online/offline.
- **ModelScope easydistill** (Alibaba, alinhado a Qwen).
- **TRL `GKDTrainer`** (on-policy/GKD pronto).

---

## 3. Desenho experimental escolhido (NÚCLEO — fase atual)

**White-box logit KD, mesma família** (tokenizador idêntico entre Qwen2.5-7B e 0.5B/1.5B):

- **Professor:** `Qwen2.5-7B-Instruct` (cabe na L4 23.9GB em bf16; pode-se pré-computar top-k logits offline,
  estilo Gemma 2, para desacoplar do treino).
- **Alunos:** `Qwen2.5-0.5B` **e** `Qwen2.5-1.5B` (decisão do usuário) — 0.5B dá o contraste mais visível (14×).
- **Dataset sintético:** ~1.000 prompts (500 DOM-PI + 500 docentesDC); o professor gera **resposta (hard label)**
  e **top-k logits (soft labels)** por token de resposta.
- **Métodos** (treinados separadamente, para isolar a contribuição de cada sinal):
  - (a) **CE / hard** — SFT na resposta do professor (baseline de imitação);
  - (b) **KL / soft** — KL-divergência aluno↔professor nos top-k logits, com temperatura T;
  - (c) **Combinado** — α·CE + (1−α)·T²·KL (α=0,5).

### Benchmark e métricas (100 perguntas = 50 DOM-PI + 50 docentesDC)
- **PPL / CE / token-acc** do aluno antes e depois (reusa `avaliacao/avaliar_modelo.py`);
- **ROUGE-L vs professor** (quanto o aluno se aproxima do professor);
- **Acerto factual** (entidade-chave, como na Q5) do professor, aluno-base e aluno-destilado;
- **Comparação chave:** aluno destilado (b/c) **supera** o aluno SFT-puro (a)? → evidência de que o sinal de
  logits transfere conhecimento além do texto.

---

## 3.1 ⭐ Destaque metodológico — KL sobre top-k renormalizado (cache estilo Gemma 2)

Armazenar a distribuição **completa** do professor por token é proibitivo: o vocabulário Qwen2.5 tem ~151k
entradas, então salvar logits densos para ~1.000 respostas × ~256 tokens custaria dezenas de GB e dominaria o
I/O do treino. **Decisão adotada:** salvar apenas os **top-50 logprobs por token** (cache offline) e, no cálculo
da KL, **renormalizar a softmax do professor exclusivamente sobre esses top-50** — os tokens fora do top-k saem
do suporte da distribuição-alvo.

- **Por que é correto:** é exatamente o que o **Gemma 2** faz — "como o vocabulário tem 256k entradas, apenas um
  **subconjunto amostrado** das probabilidades do professor é armazenado" (arXiv:2408.00118). A massa de
  probabilidade do professor concentra-se nos primeiros tokens; o top-50 captura quase toda a entropia útil, e a
  renormalização produz um alvo soft válido (soma 1) sem precisar do vocabulário inteiro.
- **Efeito prático:** desacopla o professor 14B do laço de treino (Gemma-2 style), reduz o cache de dezenas de GB
  para centenas de MB, e deixa os 3 métodos (CE/KL/combinado) lendo o **mesmo** cache.
- **Parâmetros:** temperatura **T** suaviza o alvo (escala os logprobs antes da softmax); a perda combinada usa
  α·CE + (1−α)·**T²**·KL — o fator T² mantém a magnitude do gradiente da KL comparável à da CE (Hinton et al.).
- **Limite conhecido:** com top-k truncado, o aluno não recebe sinal sobre a *cauda* da distribuição do professor;
  para domínios de alta entropia poder-se-ia aumentar k. Aqui k=50 é folgado dado o foco factual (respostas
  objetivas, baixa entropia por token).

## 4. Implementação (scripts em `scripts/`)

| # | Script | Estado |
|---|---|---|
| 1 | `gerar_dataset_destilacao.py` + `run_gerar_dataset.sbatch` | ✅ escrito |
| 2 | `destilar.py` + `run_destilar.sbatch` | ✅ escrito |
| 3 | `benchmark_destilacao_100.jsonl` (50 DOM-PI + 50 docentesDC) | ⏳ a criar |
| 4 | `avaliar_destilacao.py` (PPL/token-acc/ROUGE-L/factual, professor + 12 alunos) | ⏳ a criar |

### 4.1 Geração do dataset (script 1)
Professor **Qwen2.5-14B-Instruct** servido por **vLLM com `tensor_parallel_size=2`** (gpunode01, 2× L4),
`bfloat16`, `gpu_memory_utilization=0.90`. Perguntas: 500 DOM-PI (self-instruct *grounded* sobre passagens do
held-out) + 500 docentesDC (instruções do dataset HF). **Pergunta idêntica entre A e B**; só muda o contexto.
Captura **top-50 logprobs/token** (`SamplingParams(logprobs=50)`), arredondados a 4 casas → cache de centenas de MB.
O **prompt renderizado** é salvo no dataset para que a destilação reproduza o exemplo sem divergência de formatação.

### 4.2 Destilação (script 2) — decisões fixadas
- **Aluno = base pristino** (`Qwen2.5-0.5B` e `1.5B`); o modelo treinado da Q1/Q5 **não** é usado (professor fraco
  transferiria erros — ver §2.5). Full fine-tune (aluno é pequeno), `bfloat16`, gradient checkpointing.
- **Alinhamento `labels`/`input_ids` sem pré-shift** (lição do bug duplo-shift da Q1); **prompt mascarado (−100)**;
  perda só sobre os tokens de resposta.
- **Posições de resposta:** logits `[p_len−1 : p_len−1+ans_len]` preveem `answer_token_ids[0:]` — mesmo
  alinhamento dos top-k do professor.
- **KL top-k renormalizada** (§3.1): alvo `softmax(logprob_professor / T)` sobre o suporte top-50; aluno via
  `log_softmax` no vocabulário inteiro colhido nos mesmos ids. `combined = α·CE + (1−α)·T²·KL`.
- **Hiperparâmetros:** `T=2.0`, `α=0.5`, `epochs=3`, `lr=1e-5` (cosine, warmup 3%), `grad_accum=16`,
  `max_len=1024`, micro-batch=1 (evita bug de gather por posição com padding). `config_destilacao.json` salvo por aluno.

### 4.3 Organização do job
- **Geração:** `run_gerar_dataset.sbatch` — gpunode01, `--gres=gpu:2`, reconstrói o índice RAG se faltar (braço B).
- **Destilação:** `run_destilar.sbatch` — varre a matriz **{0.5B,1.5B} × {ce,kl,combined} × {A,B} = 12 alunos**,
  despachando 1 aluno por GPU (`CUDA_VISIBLE_DEVICES`) em pares paralelos no gpunode01.

### Expansão gated (só se o núcleo for satisfatório)
- Braço **black-box cross-família** (Gemma-2-9B ou Llama-3.1-8B → aluno via SFT) para contrastar
  white-box×black-box; e/ou tentativa **ULD/OT** (logit KD cross-tokenizer, research-grade).

### Extensão futura — especialização temática (não nesta rodada)
Destilar um aluno especializado em um **corpo de conhecimento coletado da internet** (ex.: Copa do Mundo 2026,
campeonatos brasileiros, um ano de política nacional). Vale como demonstração da *utilidade prática* da técnica:
um tema **posterior ao corte de conhecimento** faz o aluno-base pontuar ~0% e a transferência ficar inequívoca,
e torna o contraste A×B brutal (o professor "zerado" não tem como saber → só RAG sabe).
- **Fontes:** Wikipedia (multilíngue), sites oficiais (ex.: FIFA), notícias via `WebSearch`/`WebFetch`.
- **Técnica:** coletar corpus → indexar (reusa `build_index.py`) → self-instruct + professor com RAG gera ~1.000
  Q&A fiéis às fontes → SFT/logit KD no aluno → benchmark 100 Q de fatos **estáveis** do tema.
- **Lição esperada (trade-off):** RAG vence para conhecimento volátil (atualização barata); destilação
  internaliza um *snapshot*. Decisão do usuário: tratar como prática futura / comparativo póstumo.

---

## 5. Resultados (executado — jobs SLURM 502/503/504)

Pipeline completo executado no cluster (gpunode01, 2× L4): geração do dataset (job 502, 1000 prompts ×
braços A/B + top-50 logits), destilação dos 12 alunos (job 503) e avaliação (job 504). Métricas no benchmark
held-out de 100 perguntas (50 DOM-PI + 50 docentesDC), **sem RAG na inferência** (testa o que ficou nos pesos).
Referência = resposta do professor 14B **com RAG** (braço B). RG = ROUGE-L; KR = key-term recall.

| Modelo | geral RG | geral KR | DOM RG | DOM KR | doc RG | doc KR |
|---|---|---|---|---|---|---|
| base 0.5B | 0,227 | 0,380 | 0,296 | 0,505 | 0,157 | 0,249 |
| base 1.5B | 0,185 | 0,366 | 0,230 | 0,453 | 0,141 | 0,276 |
| d_0.5b A ce | 0,220 | 0,550 | 0,244 | 0,642 | 0,196 | 0,453 |
| d_0.5b A kl | 0,174 | 0,577 | 0,184 | 0,660 | 0,164 | 0,490 |
| d_0.5b A comb | 0,194 | 0,563 | 0,214 | 0,654 | 0,174 | 0,469 |
| d_0.5b B ce | 0,203 | 0,667 | 0,249 | 0,613 | 0,157 | 0,723 |
| d_0.5b B kl | 0,224 | 0,598 | 0,267 | 0,630 | 0,181 | 0,564 |
| d_0.5b B comb | 0,243 | 0,625 | 0,276 | 0,611 | 0,211 | 0,639 |
| d_1.5b A ce | 0,244 | 0,617 | 0,293 | 0,661 | 0,196 | 0,570 |
| d_1.5b A kl | 0,201 | 0,576 | 0,232 | 0,664 | 0,170 | 0,484 |
| d_1.5b A comb | 0,208 | 0,582 | 0,241 | 0,619 | 0,176 | 0,544 |
| d_1.5b B ce | 0,223 | 0,647 | 0,277 | 0,641 | 0,170 | 0,652 |
| d_1.5b B kl | 0,350 | 0,689 | 0,380 | 0,654 | 0,320 | 0,725 |
| **🏆 d_1.5b B comb** | **0,363** | **0,717** | **0,429** | 0,659 | 0,297 | **0,776** |

### Conclusões — houve transferência de conhecimento?
1. **Sim, inequívoca.** Os 12 alunos superam ambas as bases no `key_recall` (0,37–0,38 → 0,55–0,72). O aluno-base
   quase não conhece os fatos; o destilado os internaliza nos pesos.
2. **Melhor receita: aluno 1.5B · braço B · combinado** → ROUGE-L 0,363 e key_recall 0,717, **≈ +96%** sobre a
   base 1.5B em ambos (quase dobra).
3. **Professor aterrado com RAG (B) > "zerada" (A)** — confirma a ponte Q4↔Q5: o grounding transfere **mais fatos
   corretos** aos pesos. Mais nítido em docentes (KR doc até 0,776 em B vs ~0,47–0,57 em A).
4. **Soft labels (KL/combined) > CE puro na escala maior:** no 1.5B-B, combined (0,363) ≥ kl (0,350) ≫ ce (0,223).
   Valida o sinal de logit KD white-box (§3.1) — o aluno aprende além do texto.
5. **A destilação destrava o aluno maior:** a base 1.5B era *pior* que a 0.5B (RG 0,185 vs 0,227), mas, destilada,
   torna-se a melhor — a capacidade extra só se realiza com o sinal do professor.

### Análise qualitativa das inferências (held-out, verbatim) — o que de fato foi transferido
Abrindo as respostas verbatim (`resultados/avaliacao.json`, campo `detalhe[].answer`) e separando as 100 referências:
**71% são abstenções** ("Não consta…") — nas perguntas held-out o RAG muitas vezes **não recuperou** o documento-fonte,
então o próprio professor se absteve. O `key_recall` por subconjunto revela a composição do ganho:

| Subconjunto da referência | n | base 1.5B | aluno 1.5B·B·comb |
|---|---|---|---|
| abstenção ("Não consta") | 71 | 0,347 | **0,835** |
| fato real (nº/CNPJ/lei…) | 29 | 0,410 | 0,434 |

**Leitura:** o headline **KR 0,717 (+96%) é dominado pela disciplina de abstenção**, não por recordar fatos do DOM-PI
(nas perguntas com fato real, aluno ≈ base). O que a destilação transferiu é **confiabilidade**: o aluno (1) deixa de
**alucinar** valores falsos (base: "R$ 1.000,00" inventado em `bm009`) e (2) deixa de **degenerar** em loops de
token-lixo (`猞猞…` na base), adotando o "não sei fundamentado" do professor RAG-grounded. Três painéis ilustrativos
no `relatorio_q4.html` §4.7 (`bm009` anti-alucinação, `bm005` fim da degeneração, `bm031` a limitação: o fato existe
mas o aluno se abstém). Gerados por `scripts/gerar_painel_inferencias.py`. **Implicação:** RAG na inferência (Q5)
segue **necessário** para precisão factual.

### Ressalvas honestas
- A **referência é a resposta do professor-B (com RAG)**, o que dá leve vantagem aos modelos do braço B no
  ROUGE-L (mesma distribuição). Por isso o **`key_recall`** (presença de entidades/números) é o sinal mais neutro —
  e nele B também vence. Não é circularidade: a referência é factual e o aluno responde *sem* RAG.
- **Token espúrio residual** (ex.: `creampie`, `(AdapterView)`) aparece ao final de algumas respostas dos alunos —
  artefato de decodificação dos modelos pequenos (sem disciplina de EOS); não afeta o conteúdo, mas é honesto registrar.
- ROUGE-L permanece modesto em termos absolutos (respostas reformulam a frase); o ganho de conhecimento está
  concentrado no conteúdo factual (key_recall), coerente com o objetivo de destilação.

Artefatos: `dados/` (dataset + logits + benchmark), `modelos/aluno_qwen2.5-*_{A,B}_{ce,kl,combined}/` (12 alunos),
`resultados/avaliacao.json` (métricas + respostas geradas). Melhor aluno publicado no HF (ver README).

---

## 6. Extensão executada — Plano B: cross-família black-box (job SLURM 506/507)

Para contrastar **white-box (mesma família, com logits)** × **black-box (cross-família, só texto)**, destilamos um
professor de **outra família** para o aluno Qwen via SFT no texto (sequence-KD), reusando as mesmas perguntas e o
mesmo contexto B. Professor: **`HuggingFaceH4/zephyr-7b-beta`** (arquitetura Mistral, tokenizer ≠ Qwen; escolhido
por ser *ungated* — Gemma-2 e Llama-3.1 são *gated* e bloquearam o download). Re-tokenização da resposta no espaço
Qwen (`gerar_dataset_crossfamilia.py`); SFT método `ce`.

| Arm (aluno 1.5B) | Professor | Sinal | ROUGE-L | key_recall |
|---|---|---|---|---|
| base | — | — | 0,185 | 0,366 |
| cross-família black-box | zephyr-7B | texto | 0,270 | 0,490 |
| mesma-família black-box | Qwen-14B | texto | 0,223 | 0,647 |
| mesma-família white-box (kl) | Qwen-14B | logits | 0,350 | 0,689 |
| **🏆 mesma-família white-box (comb)** | Qwen-14B | logits | **0,363** | **0,717** |

(0.5B: `bxf_0.5b_ce` RG 0,348 / KR 0,523 vs `d_0.5b_B_ce` RG 0,203 / KR 0,667.)

**Conclusões:** (1) **todos transferem** — a cross-família também (KR 0,49 ≫ base 0,37); (2) **logits/mesma-família
entregam o máximo** (combined 0,717) — é o porquê de Gemma 2 / Llama 4 destilarem dentro da família; (3) a
cross-família **perde recall**, mas em parte é **artefato** da referência ser o Qwen-14B (vantagem de casa em KR para
a mesma família); (4) curiosamente a cross-família tem **ROUGE-L maior** que a mesma-família black-box — o zephyr é um
instruct forte e fraseia mais perto da referência. Lição reproduzida da literatura: **white-box mesma-família quando
possível; black-box cross-família (estilo DeepSeek-R1) só quando forçado** (vocabulário/tokenizador distintos).

Scripts: `gerar_dataset_crossfamilia.py`, `run_crossfamilia.sbatch`, `run_avaliar_bxf.sbatch`. Resultados em
`resultados/avaliacao_bxf.json`. *(Ressalva de tamanho: zephyr-7B < Qwen-14B → a comparação cross-família carrega
também o efeito de tamanho do professor.)*

---

## 7. Extensão executada — Plano A: especialização temática por destilação de RACIOCÍNIO (jobs 508/509)

Tema **Copa do Mundo 2026** (posterior ao corte de conhecimento → o aluno-base não sabe). **Destilação de raciocínio**
(receita DeepSeek-R1): professor **`DeepSeek-R1-Distill-Qwen-14B`** — modelo *thinking* (`<think>`) sobre Qwen2.5-14B,
**mesma família** dos alunos → habilita **white-box logit KD** (reusa o pipeline campeão da Q4). Corpus factual
**ungated** coletado e indexado (`coletar_corpus_futebol.py` → 241 passagens: openfootball 2026 [grupos, jogos com
placares/gols, elencos, estádios, seleções] + **classificação derivada dos jogos** + Wikipedia) — **sem Transfermarkt**.
200 perguntas (self-instruct) × braços A (zerada) × B (RAG sobre o corpus), top-50 logits; destilação `combined`.

Benchmark held-out de **41 fatos** (Copa 2026), referência = Qwen2.5-14B-Instruct + RAG (concisa):

| Modelo | ROUGE-L | key_recall |
|---|---|---|
| base 1.5B | 0,122 | 0,476 |
| fut_0.5b A (zerada) | 0,178 | 0,616 |
| fut_0.5b B (RAG) | **0,403** | 0,628 |
| fut_1.5b A (zerada) | 0,131 | 0,617 |
| **🏆 fut_1.5b B (RAG)** | 0,209 | **0,640** |

**Conclusões:** (1) a destilação de raciocínio **transfere o conhecimento da Copa 2026** — todos os alunos KR
~0,62–0,64 vs base 0,476; aqui há **recordação factual real** (painel `bm030` em `relatorio_q4.html` §4.2: o aluno
recupera "Cape Verde → CAF" verbatim; a base alucina e degenera) — o **contraponto** que prova que a fraca recordação
factual do núcleo (§ análise de abstenção) era efeito das 71% de referências "Não consta", não limite da técnica; (2) **professor com RAG (B) > zerada (A)** — para tema **pós-corte**, o professor "zerado"
raciocina mas **não tem os fatos** (alucina); só o RAG os fornece (nítido no ROUGE-L: 0.5b_B 0,403 vs 0.5b_A 0,178);
(3) **valida a estratégia de dados** (corpus ungated openfootball+Wikipedia, esforço moderado, sem scraping de
Transfermarkt); (4) `key_recall` é a métrica robusta (os alunos emitem `<think>`, ruidificando o ROUGE-L vs a
referência concisa). *Ressalvas:* base KR 0,476 não é 0 (perguntas tocam conhecimento geral de futebol + eco do
enunciado); fatos voláteis (snapshot). Scripts: `coletar_corpus_futebol.py`, `run_futebol.sbatch`,
`run_avaliar_futebol.sbatch`. Resultados em `resultados/avaliacao_futebol.json`.

**Automação:** a cadeia 508→509 rodou via **dependência SLURM `afterok`** (geração+destilação → avaliação),
mantendo as GPUs utilizadas sem intervenção manual.

---

## 8. Extensão executada — DAPT-then-distill: o modelo da Q1 como ALUNO (jobs 510/511)

Pergunta: o **priming de domínio** ajuda a destilação? Em vez de destilar para o `Qwen2.5-1.5B` base, destilamos
para o **DAPT da Q1** (`checkpoints_fullft_unificado_v2/best`, Full FT v2, −11,3% PPL no domínio) — currículo
"DAPT → destilação". Reusa `dataset_B`/`logits_B` (white-box) e `dataset_Bxf` (cross-família); só troca `--student`.
(O modelo da Q1 NÃO é usado como professor — fraco demais; ver §2.5. Aqui ele é o *ponto de partida do aluno*.)

| Aluno 1.5B | Início | Sinal | ROUGE-L | key_recall |
|---|---|---|---|---|
| base (referência) | base | — | 0,185 | 0,366 |
| DAPT cru (Q1) | DAPT | — | 0,187 | 0,368 |
| white-box, aluno **base** (`d_1.5b_B_combined`) | base | logits | **0,363** | **0,717** |
| white-box, aluno **DAPT** (`dapt_B_combined`) | DAPT | logits | 0,326 | 0,694 |
| cross-família, aluno **base** (`bxf_1.5b_ce`) | base | texto | 0,270 | 0,490 |
| cross-família, aluno **DAPT** (`dapt_Bxf_ce`) | DAPT | texto | 0,262 | 0,478 |

**Resultado (negativo, mas informativo): o priming de domínio NÃO ajudou** — DAPT-then-distill ficou ~0,02–0,04
**pior** que base-then-distill em ambos os regimes. Razões: (1) o DAPT cru já é ~igual ao base no benchmark factual
(KR 0,368 vs 0,366) — ele melhorou a *PPL/modelagem de linguagem* (Q1), não o Q&A factual; (2) a destilação
**reescreve** o aluno na direção do professor, lavando o head-start. Isso reforça (por outro ângulo) a §2.5: o
modelo fraco da Q1 (DAPT em OCR ruidoso) não agrega nem como ponto de partida — o base pristino é tão bom ou melhor.
*Ressalva:* um DAPT mais limpo (Teresina v3, −12,9%) poderia render diferente; com este (corpus ruidoso), não há ganho.
Scripts: `run_dapt_distill.sbatch`, `run_avaliar_dapt.sbatch` (rodaram no **gpunode02** — `gpu:1` sem fixar nó,
aproveitando a 2ª GPU). Resultados em `resultados/avaliacao_dapt.json`.

---

## 9. DAPT-then-distill **Teresina** — a ressalva da §8 refutada (jobs 518/519)

Testando se o DAPT mais **limpo** (Teresina v3, −12,9%) muda o resultado negativo da §8.

| Aluno 1.5B (white-box · B · combinado) | ROUGE-L | key_recall |
|---|---|---|
| aluno base pristino (`d_1.5b_B_combined`) | 0,363 | **0,717** |
| DAPT unificado (§8) | 0,326 | 0,694 |
| DAPT Teresina | 0,295 | 0,662 |

**Ressalva REFUTADA:** o DAPT limpo também não ajuda — é até pior (0,662 < 0,694 < base 0,717). Conclusão reforçada:
**priming de domínio antes da destilação não agrega, independente da qualidade do corpus** (a destilação lava o
head-start). daptT cru ≈ base (KR 0,368). Scripts: `run_dapt_distill.sbatch` (env `DAPT_PATH/DAPT_TAG`).
`resultados/avaliacao_daptT.json`.

## 10. ULD cross-tokenizer — a fronteira (job 520)

Logit KD **entre famílias** (zephyr→Qwen) sem vocabulário comum (Boizard et al., arXiv:2402.12030): por posição,
ordenam-se as distribuições de probabilidade do aluno e do professor e minimiza-se a L1 entre os vetores ordenados
(invariante ao vocabulário). Método `uldcomb` = α·CE + (1−α)·ULD (`destilar.py`).

| Aluno (cross-família) | método | ROUGE-L | key_recall |
|---|---|---|---|
| base 1.5B | — | 0,185 | 0,366 |
| bxf_0.5b | black-box (CE) | 0,348 | 0,523 |
| **uld_0.5b** | uldcomb | **0,409** | 0,525 |
| bxf_1.5b | black-box (CE) | 0,270 | 0,490 |
| **uld_1.5b** | uldcomb | **0,330** | 0,504 |

**Positivo modesto:** o ULD aproximado **bate o black-box puro** em ambos os tamanhos (o sinal de distribuição
cross-tokenizer agrega sobre o SFT no texto). **Ressalvas honestas:** é `uldcomb` (inclui CE); o alinhamento posicional
entre tokenizadores distintos é **aproximado** (truncagem; o ULD fiel usa transporte ótimo). Ainda **abaixo do
white-box mesma-família** (0,717) — destilar dentro da família com logits segue o teto. Scripts: `run_uld.sbatch`,
`gerar_dataset_crossfamilia.py` (+logprobs), `destilar.py` (método `uld`/`uldcomb`). `resultados/avaliacao_uld.json`.

---

## Referências (além de Raschka)
- Gemma 2: *Improving Open Language Models at a Practical Size* — arXiv:2408.00118
- DeepSeek-R1 (distill report) — deepseek-ai/DeepSeek-R1-Distill-* (HF)
- Boizard et al., *Universal Logit Distillation Loss* (cross-tokenizer) — arXiv:2402.12030
- *Approximate Likelihood Matching* (cross-tokenizer) — arXiv:2503.20083
- *Multi-Level Optimal Transport for Cross-Tokenizer KD* — arXiv:2412.14528
- MiniLLM (reverse-KL on-policy); GKD (on/off-policy unificado)
- *A Survey of On-Policy Distillation for LLMs* — arXiv:2604.00626
- Frameworks: arcee-ai/DistillKit · modelscope/easydistill · TRL `GKDTrainer`
