# Relatório — Questão 4: Destilação de Conhecimento

## Status: ✅ Concluída (estudo + 12 alunos destilados + avaliação; melhor = 1.5B · braço B · combinado, +96%)

## 1. O que a questão pede

Investigar como se faz **destilação de conhecimento** entre modelos de linguagem, definir um **professor** e um
**aluno**, gerar um **dataset sintético** para transferir conhecimento do professor para o aluno, montar um
**benchmark de 100 perguntas** e, com ele, medir o professor e o aluno **antes e depois** — respondendo à pergunta
central: **houve transferência de conhecimento?**

---

## 2. Conceitos básicos (para ler o resto sem tropeçar)

A destilação parte de uma ideia simples: um modelo **grande e forte** (o **professor**) "ensina" um modelo
**pequeno** (o **aluno**) a se comportar como ele, mas a uma fração do custo de inferência. O aluno não vê o
mundo todo de novo — ele aprende a **imitar as saídas do professor**.

Há duas formas de o professor "ensinar", e a diferença é o coração desta questão:

- **Hard label (rótulo duro):** o professor escreve uma resposta em **texto**, e o aluno é treinado a reproduzir
  exatamente aquele texto. É o aprendizado por imitação literal — equivale a um *fine-tuning supervisionado* (SFT)
  usando o professor como gerador de gabaritos.
- **Soft label (rótulo suave):** além do texto, o professor expõe, **para cada token**, a sua **distribuição de
  probabilidade** sobre todo o vocabulário — ou seja, não só "a próxima palavra é X", mas "era 70% X, 20% Y, 5% Z…".
  Essa distribuição carrega o *como o professor pensa*: quais alternativas ele considerou e com que confiança.

> **Ligação com Raschka:** no livro, a saída do modelo antes da decisão final é o vetor de **logits**, convertido em
> probabilidades pela **softmax**; o treino minimiza a **cross-entropy** entre a previsão e o alvo. Na destilação por
> soft label, o "alvo" deixa de ser um único token (vetor *one-hot*) e passa a ser **a distribuição inteira do
> professor** — o aluno aprende a curva, não só o pico.

Três termos técnicos que aparecem o tempo todo:

| Termo | Significado direto |
|---|---|
| **Logit** | pontuação bruta que o modelo dá a cada token antes da softmax (Raschka, cap. de geração de texto). |
| **CE (cross-entropy)** | perda que treina o aluno a reproduzir o **texto** do professor (sinal *hard*). |
| **KL (divergência de Kullback–Leibler)** | perda que aproxima a **distribuição** do aluno da distribuição do professor (sinal *soft*). |
| **Temperatura (T)** | fator que "achata" a distribuição do professor antes da softmax: com T alto, as alternativas secundárias ganham peso e o aluno enxerga o raciocínio completo, não só o token vencedor. |

E duas categorias de método, definidas por **o que o aluno enxerga do professor**:

- **White-box ("caixa branca"):** o aluno usa os **logits** internos do professor. Isso **só funciona se professor
  e aluno compartilharem o mesmo tokenizador/vocabulário** — caso contrário, "70% no token nº 4012" não significa a
  mesma coisa para os dois. Na prática, exige **mesma família de modelos**.
- **Black-box ("caixa preta"):** o aluno só vê o **texto** que o professor produziu (não os logits). Funciona entre
  **qualquer** par de modelos, mas joga fora o sinal mais rico (as probabilidades).

---

## 3. Investigação: como a indústria faz destilação

Pesquisando como os modelos de produção recentes são destilados, três padrões organizam todo o nosso desenho:

**3.1 A prática consolidada é destilar dentro da mesma família.** Os principais modelos abertos pequenos são
versões destiladas de irmãos maiores da mesma linhagem (mesmo tokenizador), justamente para poder usar
**white-box logit KD**. Quando se destila *entre* famílias diferentes, o caminho usado é o **black-box** (treinar o
aluno apenas no texto do professor) — foi assim, por exemplo, que modelos de raciocínio foram comprimidos em alunos
de famílias variadas: gerou-se um grande volume de respostas e treinou-se por SFT, **sem logits**.

**3.2 Os três regimes técnicos e seus custos:**

| Regime | Mesmo tokenizador? | Custo | Maturidade |
|---|---|---|---|
| **White-box logit KD** (KL sobre os logits) | **Sim** → só mesma família | guardar o professor em memória ou pré-computar os logits | padrão da indústria |
| **Black-box sequence-KD** (SFT no texto do professor) | Não → qualquer família | só o custo de gerar o dataset sintético | padrão e robusto |
| **Cross-tokenizer logit KD** (logits entre famílias, por aproximação) | Não (aproximado) | alto, ainda experimental | fronteira de pesquisa |

**3.3 Onde o aluno "treina":** treinar no texto pronto do professor sofre de um descompasso entre treino e uso real
(o aluno nunca vê os próprios erros durante o treino). Técnicas mais novas destilam **na distribuição gerada pelo
próprio aluno** — mais fiéis, porém mais caras de implementar (exigem o aluno gerando texto em pleno treino). Aqui
ficamos no regime mais simples e bem estabelecido.

**3.4 O que descartamos de propósito:**
- **Cross-modalidade** (imagem/voz → texto): fora de escopo — o corpus é texto governamental puro.
- **Cross-especialidade** (ex.: um professor de código ensinando sobre administração pública): descartado — um
  professor de domínio incompatível não tem o que transferir e tende a *degradar* o aluno.
- **Cross-família**: viável apenas como braço **black-box**, reservado como **extensão** caso o núcleo desse certo.

**3.5 Por que NÃO usar o modelo da Q1/Q5 como professor.** O princípio da destilação é *professor ≫ aluno*. O nosso
modelo DAPT da Q1 é pequeno (1.5B) e foi treinado sobre um corpus com OCR corrompido — usá-lo como professor
transferiria os próprios erros para o aluno. Por isso o professor é um **modelo oficial forte e factualmente limpo**
da mesma família (`Qwen2.5-Instruct`, que compartilha o tokenizador dos alunos e habilita o white-box). O aluno é o
**modelo base "pristino"** (sem treino prévio), que serve de ponto de partida neutro para *medir* o ganho.

**3.6 O grounding do professor como variável de estudo.** Um professor genérico não conhece os fatos específicos do
DOM-PI e, perguntado "na seco", **aluciná**. Então transformamos o *acesso a fatos* do professor numa variável
controlada — os **braços A e B**:

| Braço | O professor responde… | O que mede |
|---|---|---|
| **A — "zerada"** | só da memória interna (sem buscar nada) | quanto o professor sabe sozinho (com alucinação) e quanto disso o aluno absorve |
| **B — com RAG** | com o contexto recuperado do índice DOM-PI (o sistema da Q5) | transferência de conhecimento **factual e correto** para os pesos do aluno |
| **C — RAG na inferência** | não destila; busca os fatos na hora de responder (baseline da Q5) | o teto: conhecimento **externo** (Q5) vs **internalizado** nos pesos (Q4) |

A pergunta-guia vira: **quanto dá para "assar" conhecimento dentro dos pesos do aluno (B) e quão perto isso chega de
simplesmente carregá-lo de fora na hora da resposta (C)?** — é a ponte direta entre a Q4 e a Q5.

---

## 4. Desenho experimental do núcleo

**White-box logit KD, mesma família** (tokenizador idêntico entre o professor Qwen2.5 e os alunos 0.5B/1.5B).

- **Professor:** `Qwen2.5-14B-Instruct` — forte, limpo e da mesma família (habilita o uso dos logits).
- **Alunos:** `Qwen2.5-0.5B` **e** `Qwen2.5-1.5B`, ambos base "pristino". O 0.5B dá o contraste mais visível
  (dezenas de vezes menor que o professor — se transferir para ele, fica inequívoco).
- **Dataset sintético:** ~1.000 prompts (500 sobre o DOM-PI + 500 sobre o *docentesDC*). Para cada prompt o
  professor gera a **resposta em texto** (hard label) e os **top-50 logits por token** (soft label).

### Por que 12 alunos? — um desenho fatorial

Não é repetição: treinamos uma **matriz controlada** para isolar a contribuição de cada fator. São três eixos
multiplicados:

**{0.5B, 1.5B}** × **{CE, KL, Combinado}** × **{Braço A, Braço B}** = **12 alunos**

| Eixo variado | Opções | A pergunta que esse eixo responde |
|---|---|---|
| **Tamanho do aluno** | 0.5B · 1.5B | a capacidade extra do aluno maior se realiza com a destilação? |
| **Sinal de treino** | CE (texto) · KL (logits) · Combinado | os **logits** transferem algo **além do texto**? |
| **Grounding do professor** | A (zerada) · B (com RAG) | aterrar o professor em fatos transfere conhecimento **correto**? |

Só com a matriz inteira é possível afirmar, sem ambiguidade, frases como "KL supera CE no 1.5B" ou "B supera A":
muda-se **um fator por vez**. As três perdas treinadas separadamente:

- **(a) CE / hard** — SFT na resposta do professor (imitação do texto, baseline);
- **(b) KL / soft** — aproxima a distribuição do aluno da do professor nos top-50 logits, com temperatura **T**;
- **(c) Combinado** — `α·CE + (1−α)·T²·KL` (α = 0,5), usando os dois sinais ao mesmo tempo.

### Benchmark e métricas (100 perguntas = 50 DOM-PI + 50 docentesDC)

- **PPL / CE / token-accuracy** do aluno antes e depois (perplexidade, cross-entropy e acerto por token — as
  métricas de modelagem de linguagem de Raschka);
- **ROUGE-L (RG)** — o quanto o **texto** do aluno se sobrepõe ao do professor;
- **key_recall (KR)** — a fração de **entidades/números-chave** (leis, CNPJ, valores) que o aluno acertou; é o
  sinal **mais neutro**, porque mede fato, não fraseado;
- **Comparação-chave:** o aluno destilado com soft label (b/c) **supera** o aluno SFT-puro (a)? Se sim, é evidência
  de que o sinal dos logits transfere conhecimento **além do texto**.

### ⭐ Destaque: KL sobre os top-50 logits (cache enxuto)

Guardar a distribuição **completa** do professor por token é inviável: o vocabulário do Qwen2.5 tem ~151 mil
entradas, e salvar isso para ~1.000 respostas × ~256 tokens custaria dezenas de GB e dominaria o I/O do treino. A
solução adotada (a mesma de modelos de produção): salvar só os **top-50 logits por token** e, no cálculo da KL,
**renormalizar a softmax do professor apenas sobre esses 50** — os demais saem do suporte da distribuição-alvo.

- **Por que é válido:** a massa de probabilidade do professor concentra-se nos primeiros tokens; o top-50 captura
  quase toda a informação útil, e a renormalização produz um alvo "soft" legítimo (soma 1) sem o vocabulário
  inteiro. Reduz o cache de dezenas de GB para **centenas de MB** e desacopla o professor pesado do laço de treino.
- **Papel da temperatura:** **T** suaviza o alvo (escala os logits antes da softmax); na perda combinada o fator
  **T²** mantém a magnitude do gradiente da KL comparável à da CE — sem ele, ajustar T desbalancearia as duas perdas.
- **Limite conhecido:** com top-k truncado, o aluno não recebe sinal sobre a *cauda* da distribuição. Para domínios
  de alta entropia conviria aumentar k; aqui, com respostas factuais (baixa entropia por token), 50 é folgado.

---

## 5. Implementação

| # | Script | Papel |
|---|---|---|
| 1 | `gerar_dataset_destilacao.py` (+ sbatch) | o professor gera respostas e top-50 logits para os ~1.000 prompts |
| 2 | `destilar.py` (+ sbatch) | treina cada aluno com CE / KL / combinado |
| 3 | `benchmark_destilacao_100.jsonl` | 50 perguntas DOM-PI + 50 docentesDC |
| 4 | `avaliar_destilacao.py` | PPL / token-acc / ROUGE-L / key_recall do professor e dos 12 alunos |

**Geração (script 1):** professor servido com inferência paralela em 2 GPUs (`bfloat16`); 500 perguntas DOM-PI
(geradas a partir de passagens do held-out) + 500 do *docentesDC*. **A pergunta é idêntica entre A e B** — só muda o
contexto fornecido. Captura os **top-50 logits/token**. O prompt já renderizado é salvo para a destilação reproduzir
o exemplo sem divergência de formatação.

**Destilação (script 2) — decisões fixadas:**
- Aluno = **base pristino** (`Qwen2.5-0.5B` e `1.5B`); fine-tuning completo (o aluno é pequeno), `bfloat16`,
  *gradient checkpointing* para caber na GPU.
- **`labels` alinhados a `input_ids` (sem pré-shift)** — a lição do bug de duplo deslocamento da Q1 — e **prompt
  mascarado (−100)**: a perda recai só sobre os tokens da **resposta**.
- **KL top-k renormalizada:** alvo `softmax(logit_professor / T)` sobre o suporte top-50; aluno via `log_softmax`
  nos mesmos ids. `combinado = α·CE + (1−α)·T²·KL`.
- **Hiperparâmetros:** `T = 2,0`, `α = 0,5`, `épocas = 3`, `lr = 1e-5` (cosine, warmup 3%), `grad_accum = 16`,
  `max_len = 1024`, micro-batch = 1.

**Organização dos jobs:** um job gera o dataset (reconstruindo o índice RAG se faltar, para o braço B) e outro varre
a matriz **{0.5B,1.5B} × {ce,kl,combined} × {A,B} = 12 alunos**, despachando alunos em paralelo, um por GPU.

---

## 6. Resultados do núcleo — houve transferência?

Pipeline executado no cluster (2× GPU L4): geração do dataset, destilação dos 12 alunos e avaliação. Métricas no
benchmark held-out de 100 perguntas, **sem RAG na inferência** (testa o que ficou *nos pesos*). A referência é a
resposta do professor com RAG (braço B). RG = ROUGE-L; KR = key_recall.

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

**Conclusões:**
1. **Sim, houve transferência — inequívoca.** Os 12 alunos superam ambas as bases no key_recall (0,37–0,38 →
   0,55–0,72). O aluno-base quase não conhece os fatos; o destilado os internaliza nos pesos.
2. **Melhor receita: aluno 1.5B · braço B · combinado** → ROUGE-L 0,363 e key_recall 0,717, **≈ +96%** sobre a
   base 1.5B (quase dobra).
3. **Professor com RAG (B) > "zerada" (A):** confirma a ponte Q4↔Q5 — aterrar o professor transfere **mais fatos
   corretos** aos pesos. Mais nítido em docentes (KR até 0,776 em B vs ~0,47–0,57 em A).
4. **Soft labels (KL/combinado) > CE puro na escala maior:** no 1.5B-B, combinado (0,363) ≥ kl (0,350) ≫ ce (0,223).
   Valida o sinal dos logits — o aluno aprende **além do texto**.
5. **A destilação destrava o aluno maior:** a base 1.5B era *pior* que a 0.5B (RG 0,185 vs 0,227), mas, destilada,
   torna-se a melhor — a capacidade extra só se realiza com o sinal do professor.

### O que de fato foi transferido (análise das respostas)

Abrindo as respostas uma a uma, **71% das referências são abstenções** ("Não consta…") — nessas perguntas held-out
o RAG muitas vezes **não recuperou** o documento-fonte, então o próprio professor se absteve. Separando o key_recall
por subconjunto:

| Subconjunto da referência | n | base 1.5B | aluno 1.5B·B·comb |
|---|---|---|---|
| abstenção ("Não consta") | 71 | 0,347 | **0,835** |
| fato real (nº/CNPJ/lei…) | 29 | 0,410 | 0,434 |

**Leitura honesta:** o headline **+96% é dominado pela disciplina de abstenção**, não por recordar fatos do DOM-PI
(nas perguntas com fato real, aluno ≈ base). O que a destilação transferiu, nesse núcleo, foi **confiabilidade**: o
aluno (1) deixa de **alucinar** valores falsos e (2) deixa de **degenerar** em loops de token-lixo, adotando o "não
sei fundamentado" do professor. **Implicação:** o RAG na inferência (Q5) segue **necessário** para precisão factual.

### Visualizações (`resultados/figuras/`)
Os resultados em gráfico (gerados por `scripts/graficos_destilacao.py` e `scripts/comparativo_por_questao.py`):
- `barras_keyrecall_config.png` — key_recall das 14 configs, base como linha de referência (todas as 12 sobem);
- `compressao_vs_keyrecall.png` — custo-benefício: 1.5B é 9× menor e 0.5B 28× menor que o professor 14B;
- `antes_depois_dominio.png` — base→melhor aluno por domínio (+46% DOM-PI, +182% docentesDC);
- `heatmap_keyrecall.png`, `abstencao_vs_fato.png`, `box_por_metodo.png`, `delta_base_vs_melhor.png`.

### Ressalvas honestas
- A referência é o professor **com RAG**, o que dá leve vantagem no ROUGE-L aos modelos do braço B (mesma
  distribuição de fraseado). Por isso o **key_recall** (presença de entidades) é o sinal mais neutro — e nele B
  também vence. Não há circularidade: a referência é factual e o aluno responde **sem** RAG.
- Resíduo de **token espúrio** ao final de algumas respostas dos alunos pequenos (sem disciplina de EOS); não afeta
  o conteúdo, mas é honesto registrar.
- O ROUGE-L absoluto é modesto (o aluno reformula); o ganho de conhecimento concentra-se no **key_recall**, coerente
  com o objetivo de destilação.

---

## 7. Extensão — white-box × black-box (cross-família)

Para contrastar **white-box (mesma família, com logits)** × **black-box (outra família, só texto)**, destilamos um
professor de **família diferente** para o aluno Qwen via SFT no texto, reusando as mesmas perguntas e o contexto B.
Professor: `zephyr-7b-beta` (arquitetura Mistral, tokenizador ≠ Qwen) — escolhido por ser de **download livre**
(outros modelos equivalentes eram de acesso restrito). A resposta foi re-tokenizada no espaço do Qwen.

| Arm (aluno 1.5B) | Professor | Sinal | ROUGE-L | key_recall |
|---|---|---|---|---|
| base | — | — | 0,185 | 0,366 |
| cross-família black-box | zephyr-7B | texto | 0,270 | 0,490 |
| mesma-família black-box | Qwen-14B | texto | 0,223 | 0,647 |
| mesma-família white-box (kl) | Qwen-14B | logits | 0,350 | 0,689 |
| **🏆 mesma-família white-box (comb)** | Qwen-14B | logits | **0,363** | **0,717** |

**Conclusões:** (1) **todos transferem** — inclusive a cross-família (KR 0,49 ≫ base 0,37); (2) **logits + mesma
família entregam o teto** (0,717) — é exatamente por isso que a indústria destila dentro da família; (3) a
cross-família **perde recall**, em parte porque a referência é o próprio Qwen (vantagem de casa); (4) curiosamente a
cross-família tem **ROUGE-L maior** que a mesma-família black-box, porque o zephyr é um *instruct* forte e fraseia
mais perto da referência. *Ressalva de tamanho:* o zephyr-7B é menor que o Qwen-14B, então a comparação cross-família
carrega também o efeito do tamanho do professor.

---

## 8. Extensão — especialização temática (Copa do Mundo 2026)

Tema escolhido de propósito **posterior ao corte de conhecimento** dos modelos → o aluno-base **não sabe nada**, o
que torna a transferência inequívoca. Professor: um modelo de **raciocínio** (que pensa em voz alta com `<think>`),
ainda **da mesma família** dos alunos → permite reusar o pipeline white-box campeão. Coletamos um corpus factual de
fontes abertas (calendário e resultados oficiais, classificações derivadas dos jogos, enciclopédia), indexamos com o
mesmo `build_index.py` e geramos 200 perguntas nos braços A (zerada) e B (RAG).

Benchmark held-out de **41 fatos** da Copa 2026 (referência = professor + RAG):

| Modelo | ROUGE-L | key_recall |
|---|---|---|
| base 1.5B | 0,122 | 0,476 |
| fut_0.5b A (zerada) | 0,178 | 0,616 |
| fut_0.5b B (RAG) | **0,403** | 0,628 |
| fut_1.5b A (zerada) | 0,131 | 0,617 |
| **🏆 fut_1.5b B (RAG)** | 0,209 | **0,640** |

**Conclusões:** (1) a destilação **transfere o conhecimento da Copa 2026** — todos os alunos vão a KR ~0,62–0,64 vs
base 0,476; aqui há **recordação factual real**, o **contraponto** que prova que a fraca recordação do núcleo (§6)
era efeito das 71% de referências "Não consta", e não um limite da técnica; (2) **professor com RAG (B) > zerada
(A)** — para tema pós-corte, o professor "zerado" raciocina mas **não tem os fatos** (alucina); só o RAG os fornece;
(3) o **key_recall** é a métrica robusta aqui (os alunos emitem `<think>`, o que ruidifica o ROUGE-L contra a
referência concisa). *Ressalvas:* a base não chega a 0 (as perguntas tocam conhecimento geral de futebol); os fatos
são um *snapshot* (conhecimento volátil). A cadeia de jobs rodou encadeada automaticamente (geração+destilação →
avaliação).

---

## 9. Extensão — o DAPT da Q1 como ponto de partida do aluno

Pergunta: **um aluno já adaptado ao domínio (o DAPT da Q1) destila melhor?** Em vez de partir do `Qwen2.5-1.5B`
base, partimos do modelo Full FT da Q1 (−11,3% de PPL no domínio) — um currículo "DAPT → destilação". (O modelo da
Q1 **não** vira professor — fraco demais; ele é só o *ponto de partida do aluno*.)

| Aluno 1.5B | Início | Sinal | ROUGE-L | key_recall |
|---|---|---|---|---|
| base (referência) | base | — | 0,185 | 0,366 |
| DAPT cru (Q1) | DAPT | — | 0,187 | 0,368 |
| white-box, aluno **base** | base | logits | **0,363** | **0,717** |
| white-box, aluno **DAPT** | DAPT | logits | 0,326 | 0,694 |
| cross-família, aluno **base** | base | texto | 0,270 | 0,490 |
| cross-família, aluno **DAPT** | DAPT | texto | 0,262 | 0,478 |

**Resultado negativo, mas informativo: o priming de domínio NÃO ajudou** — ficou ~0,02–0,04 *pior* que partir do
base. Razões: (1) o DAPT cru já é ~igual ao base no benchmark **factual** (KR 0,368 vs 0,366) — ele melhorou a
*modelagem de linguagem* (a meta da Q1), não o Q&A; (2) a destilação **reescreve** o aluno na direção do professor,
"lavando" o head-start. Testamos também com o DAPT mais **limpo** (Teresina, −12,9%) e o resultado foi até pior
(KR 0,662). **Conclusão reforçada: o priming de domínio antes da destilação não agrega, independentemente da
qualidade do corpus.**

---

## 10. Extensão — logit KD entre tokenizadores diferentes (a fronteira)

E se quiséssemos os benefícios do white-box **mesmo entre famílias** (tokenizadores distintos)? Há uma técnica de
fronteira que torna isso possível por **aproximação**: em cada posição, ordenam-se as distribuições de probabilidade
do aluno e do professor e minimiza-se a diferença entre os **vetores ordenados** — uma medida que não depende de os
dois usarem o mesmo vocabulário. Aplicamos uma versão combinada com a CE (`uldcomb`).

| Aluno (cross-família) | método | ROUGE-L | key_recall |
|---|---|---|---|
| base 1.5B | — | 0,185 | 0,366 |
| bxf_0.5b | black-box (CE) | 0,348 | 0,523 |
| **uld_0.5b** | uldcomb | **0,409** | 0,525 |
| bxf_1.5b | black-box (CE) | 0,270 | 0,490 |
| **uld_1.5b** | uldcomb | **0,330** | 0,504 |

**Positivo modesto:** a aproximação **bate o black-box puro** nos dois tamanhos — o sinal de distribuição
cross-tokenizer agrega algo sobre o SFT no texto. **Ressalvas:** inclui a CE na perda; o alinhamento posicional entre
tokenizadores distintos é **aproximado**; e ainda fica **abaixo do white-box mesma-família** (0,717) — destilar
dentro da família com logits continua sendo o teto.

---

## 11. Como nos comparamos com a literatura

A "manchete" canônica de destilação cruza **retenção** (quanto do professor o aluno mantém) com **compressão**
(quantas vezes menor). O caso histórico é o **DistilBERT** — ~97% do BERT com −40% de parâmetros e +60% de
velocidade. Toolkits e relatórios recentes (DistilQwen na própria família Qwen; modelos pequenos destilados de irmãos
maiores) reportam o mesmo par de eixos. Posicionando nosso resultado nesse vocabulário:

| Projeto | Família | Sinal | Compressão | Métrica principal |
|---|---|---|---|---|
| DistilBERT (clássico) | BERT | white-box | ~1,7× (−40%) | ~97% do professor (GLUE) |
| DistilQwen / modelos pequenos de produção | Qwen (a nossa) | white-box / sequence | 2–14× | win-rate, tarefas (AlpacaEval/MT-Bench/IFEval) |
| **Este trabalho (núcleo)** | Qwen | white-box (top-50 logits) | **9× (1.5B) / 28× (0.5B)** | key_recall +96% vs base; ROUGE-L 0,363 |

**Leitura honesta:** as manchetes "97% do professor" usam **benchmarks públicos** (GLUE/MMLU/AlpacaEval) onde o
professor pontua <100% — então a retenção é uma fração legítima. No nosso núcleo a referência é o **próprio professor
com RAG** (100% por construção), então reportamos **compressão** (9×/28×, diretamente comparável) e **ganho sobre a
base** (+96%), deixando a **retenção ancorada** para o benchmark público (§12).

## 12. Retenção ancorada em benchmark público (executado — ENEM)

Para um "% do professor" comparável à literatura, medimos aluno e professor no **ENEM**
(`eduagarcia/enem_challenge`, 200 questões de múltipla escolha), por **log-verossimilhança da alternativa** (sem
geração; `scripts/avaliar_benchmark_publico.py`). Professor 14B em 8-bit define o teto; **retenção = acc_aluno / acc_professor**.

| Modelo | Acurácia ENEM | Retenção (% do professor) |
|---|---|---|
| professor 14B (teto) | 0,455 | 100,0% |
| base 0.5B | 0,225 | 49,5% |
| d 0.5B·B·combinado | 0,245 | **53,8%** |
| base 1.5B | 0,330 | 72,5% |
| d 1.5B·B·kl | 0,325 | 71,4% |
| d 1.5B·B·combinado | 0,315 | 69,2% |

**Leitura:** ENEM é um benchmark **geral** (nada a ver com DOM-PI/docentes), então mede o efeito colateral da
especialização. Os destilados **preservam** a capacidade da base (0.5B até melhora +4,3 pp; 1.5B fica ~1–3 pp abaixo,
dentro do ruído de 200 questões) — **sem esquecimento catastrófico**. Número ancorado: os alunos **1.5B retêm ~70%
da acurácia do professor 14B sendo 9× menores** (mesma família de afirmação do "97% do BERT" do DistilBERT; aqui o
protocolo é mais difícil — zero-shot, log-prob, sem chat template — daí o teto absoluto modesto, 45,5%).
Gráfico: `resultados/figuras/retencao_benchmark_publico.png`. Resultados: `resultados/avaliacao_benchmark_publico.json`.
Job: `scripts/run_benchmark_publico.sbatch`.

## 13. Síntese para apresentação

- **Por que 12 alunos:** desenho fatorial {tamanho} × {sinal} × {grounding} para isolar cada efeito.
- **Por que esses modelos:** professor forte e **da mesma família** (habilita logits); alunos base pequenos (0.5B
  dá o contraste máximo); o modelo fraco da Q1 não serve de professor (transferiria erros).
- **Resultado-âncora:** houve transferência inequívoca; melhor receita **1.5B · B · combinado, +96%**; **logits >
  texto** e **professor com RAG > zerada**.
- **Honestidade:** no núcleo, o ganho foi sobretudo **confiabilidade** (parar de alucinar/degenerar); a extensão da
  Copa 2026 demonstra **recordação factual real**; e o RAG na inferência (Q5) segue necessário para precisão.

## Referência metodológica
- Sebastian Raschka — *Build a Large Language Model (From Scratch)*: logits e softmax na geração de texto;
  cross-entropy e perplexidade como objetivo e métrica de treino; convenção de alinhamento de rótulos (base da
  correção do bug de duplo deslocamento reaproveitada aqui na máscara de perda do aluno).
