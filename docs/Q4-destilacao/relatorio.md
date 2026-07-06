# Relatório — Questão 4: Destilação de Conhecimento

## Status: ✅ Concluída (estudo + 12 alunos destilados + avaliação; melhor = 1.5B · braço B · combinado, +96%)

## 1. O que a questão pede

Investigar como se faz **destilação de conhecimento** entre modelos de linguagem, definir um **professor** e um
**aluno**, gerar um **dataset sintético** para transferir conhecimento do professor para o aluno, montar um
**benchmark de 100 perguntas** e, com ele, medir o professor e o aluno **antes e depois** — respondendo à pergunta
central: **houve transferência de conhecimento?**

### Escopo: o núcleo (exigência) × os extras

O enunciado pede a **destilação em si**: definir um professor e um aluno, gerar um dataset sintético, treinar o
aluno a imitar o professor e medir o ganho num benchmark. **Esse é o núcleo deste relatório** — e o foco da
implementação descrita nas §§ 2 a 6: o professor `Qwen2.5-14B` ensina os alunos `0.5B`/`1.5B` por três sinais de
treino (CE / KL / combinado).

Um recurso vai **além do enunciado** e fica marcado como **🧩 extra**: aterrar o professor com **RAG** (o sistema
da Q5) *antes* de ele gerar as respostas — o chamado **braço B**. A recuperação por RAG **não é exigência da Q4**;
ela entra aqui só para (a) dar ao professor os fatos corretos do DOM-PI e (b) costurar a ponte Q4↔Q5. A versão
**fiel ao enunciado é o braço A** (o professor responde apenas da própria memória, sem buscar nada). O braço B e as
demais extensões (§§ 7 a 12 — cross-família, Copa 2026, DAPT, cross-tokenizer, retenção em ENEM) são
**enriquecimentos**, não o núcleo. Detalhe da pilha de RAG: o índice da Q5 é *flat* (matriz `e5-base` normalizada +
cosseno exato em NumPy), **sem banco vetorial dedicado** — Chroma/HNSW existem só no pipeline de ingestão anterior,
fora das seis questões.

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

**3.6 🧩 Extra — o grounding do professor como variável de estudo.** *(Vai além do enunciado.)* Um professor
genérico não conhece os fatos específicos do DOM-PI e, perguntado "na seco", **alucina**. Então, como
**enriquecimento**, transformamos o *acesso a fatos* do professor numa variável controlada — os **braços A e B**:

| Braço | O professor responde… | O que mede | Papel |
|---|---|---|---|
| **A — "zerada"** | só da memória interna (sem buscar nada) | quanto o professor sabe sozinho (com alucinação) e quanto disso o aluno absorve | **núcleo** (fiel ao enunciado) |
| **B — com RAG** | com o contexto recuperado do índice DOM-PI (o sistema da Q5) | transferência de conhecimento **factual e correto** para os pesos do aluno | **🧩 extra** (usa o RAG da Q5) |
| **C — RAG na inferência** | não destila; busca os fatos na hora de responder (baseline da Q5) | o teto: conhecimento **externo** (Q5) vs **internalizado** nos pesos (Q4) | referência (é a própria Q5) |

A pergunta-guia do extra vira: **quanto dá para "assar" conhecimento dentro dos pesos do aluno (B) e quão perto isso
chega de simplesmente carregá-lo de fora na hora da resposta (C)?** — é a ponte direta entre a Q4 e a Q5. O núcleo
da questão, porém, é respondido já com o braço A: **o aluno aprende com o professor?**

**3.7 🧩 Extra — como o RAG recupera o contexto (mecanismo).** Quando o braço B é usado, o professor recebe um
**contexto** antes de responder; esse contexto vem de uma **busca por similaridade** sobre o corpus. O passo a passo:

1. **Indexação (uma vez):** o corpus é quebrado em *chunks*; cada *chunk* vira um **vetor** (*embedding*) de 768
   dimensões pelo modelo `multilingual-e5-base`, **normalizado** (norma 1). Guarda-se a matriz `N×768` (aqui
   `N ≈ 175 mil`) e os textos correspondentes.
2. **Consulta:** a pergunta é embutida pelo **mesmo** modelo (o e5 é assimétrico: prefixo `query:` na pergunta,
   `passage:` nos documentos). Como todos os vetores têm norma 1, o **produto interno** entre a pergunta e cada
   *chunk* **é o cosseno** — a medida de similaridade. Ordenam-se os *chunks* por esse escore e tomam-se os **top-k**.
3. **Uso:** os top-k *chunks* entram como contexto do professor (braço B) — ou, no v2, a própria passagem-fonte.

> **Índice *flat* × banco vetorial dedicado.** As técnicas mais conhecidas usam um **banco vetorial dedicado**
> (Chroma, FAISS, Qdrant, pgvector): ele constrói um **índice aproximado (ANN)** — tipicamente um grafo **HNSW** —
> que devolve os vizinhos mais próximos em tempo **sublinear**, além de oferecer persistência, filtros por metadados
> e atualização incremental. É o que se usa quando há **milhões/bilhões** de vetores ou consultas concorrentes. Aqui,
> como os ~175 mil vetores **cabem em RAM**, optamos pelo índice ***flat* (força bruta)**: um único **produto de
> matrizes** compara a pergunta com **todos** os vetores e um `argsort` pega os melhores. É a **mesma semântica de
> cosseno**, mas **exata** (não aproximada) e **sem infraestrutura extra** — mais simples e reproduzível para a escala
> do trabalho. Um banco dedicado passaria a valer a pena com ~10–100× mais vetores ou com escrita/consulta *online*.
> *(No repositório há também um pipeline antigo de ingestão em ChromaDB (`src/vector_db/`, HNSW/cosseno), fora das
> seis questões.)*

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

| Eixo variado | Opções | A pergunta que esse eixo responde | Onde entra |
|---|---|---|---|
| **Tamanho do aluno** | 0.5B · 1.5B | a capacidade extra do aluno maior se realiza com a destilação? | núcleo |
| **Sinal de treino** | CE (texto) · KL (logits) · Combinado | os **logits** transferem algo **além do texto**? | núcleo |
| **Grounding do professor** | A (zerada) · B (com RAG) | aterrar o professor em fatos transfere conhecimento **correto**? | 🧩 extra (B) |

**Lendo a matriz pela lente do escopo:** os **6 alunos do braço A** ({0.5B,1.5B} × {CE,KL,combinado}) são o
**núcleo exigido** — já respondem "o aluno aprende com o professor?" e "logits batem texto?" sem nenhum RAG. Os
outros **6 alunos do braço B** são o **🧩 extra**: o mesmo desenho, mas com o professor aterrado pelo RAG da Q5,
para medir se um professor *com os fatos certos* transfere conhecimento **mais correto**.

Só com a matriz inteira é possível afirmar, sem ambiguidade, frases como "KL supera CE no 1.5B" ou "B supera A":
muda-se **um fator por vez**. As três perdas treinadas separadamente:

- **(a) CE / hard** — SFT na resposta do professor (imitação do texto, baseline);
- **(b) KL / soft** — aproxima a distribuição do aluno da do professor nos top-50 logits, com temperatura **T**;
- **(c) Combinado** — `α·CE + (1−α)·T²·KL` (α = 0,5), usando os dois sinais ao mesmo tempo.

### Benchmark e métricas (100 perguntas = 50 DOM-PI + 50 docentesDC)

Duas métricas de saída, calculadas por `avaliar_destilacao.py`:

- **ROUGE-L (RG)** — o quanto o **texto** do aluno se sobrepõe ao da referência. É a **F1 sobre a maior subsequência
  comum (LCS)** de tokens normalizados (minúsculas, sem acento): mede o fraseado.
- **key_recall (KR)** — a fração de **termos-chave da referência** presentes na resposta do aluno. "Termo-chave" é
  captado por uma regex — **palavras Iniciadas em Maiúscula** (nomes, órgãos, leis) **ou números** (valores, CNPJ,
  datas) — e a checagem é por *substring* (minúsculas). É o sinal **mais neutro**, porque mede **fato**, não fraseado;
  quando a referência não tem nenhum termo-chave, a pergunta é excluída desse cálculo (não conta como 0).
- **Comparação-chave:** o aluno destilado com soft label (b/c) **supera** o aluno SFT-puro (a)? Se sim, é evidência
  de que o sinal dos logits transfere conhecimento **além do texto**.

### Como as métricas são coletadas (protocolo de avaliação)

O ponto que garante comparabilidade: **toda pergunta passa por todos os modelos, do mesmo jeito.** Para cada um dos
14 modelos (2 bases + 12 alunos), o script:

1. **percorre as 100 perguntas** do benchmark (as mesmas para todos);
2. **gera a resposta em modo `greedy`** (`do_sample=False`, `max_new_tokens=200`), com o mesmo *system prompt*
   ("assistente factual e conciso") e — crucialmente — **sem nenhum contexto/RAG na inferência**: mede-se só o que
   ficou **nos pesos** do aluno;
3. compara a resposta com a **referência fixa** daquela pergunta e calcula RG e KR;
4. agrega por **média** — geral e por domínio (DOM-PI × docentesDC, pelo campo `source`).

A **referência é a mesma para todos os modelos**: a resposta do **professor com RAG** (a melhor aproximação de
verdade factual disponível). Como benchmark, gerador e referência são idênticos entre modelos, as diferenças de RG/KR
são atribuíveis **só ao modelo** — é o que legitima frases como "o aluno X supera o aluno Y". *(Observação honesta:
esta avaliação de saída **não** mede perplexidade/token-accuracy — essas métricas de modelagem de linguagem são as da
Q1; aqui o foco é a qualidade factual da resposta gerada.)*

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
| 4 | `avaliar_destilacao.py` | ROUGE-L / key_recall do professor e dos 12 alunos (protocolo acima) |

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

### O laço de destilação, passo a passo (o coração da implementação)

Para deixar a mecânica explícita, eis o que acontece **a cada passo de treino** de um aluno, sobre um exemplo
`(pergunta → resposta_do_professor)` com os `top-50 logits/token` já em cache:

1. **Montagem do exemplo.** Concatena-se `prompt + resposta` num único `input_ids`. Os `labels` recebem **cópia
   exata** de `input_ids` (sem pré-shift — a lição do bug da Q1: o HF já desloca internamente), e a parte do
   **prompt é mascarada com −100**, de modo que a perda só conta nos tokens da **resposta**.
2. **Forward do aluno.** O aluno produz seus próprios `logits` para cada posição da resposta.
3. **Perda CE (sinal *hard*, do texto).** Cross-entropy entre os `logits` do aluno e o **token real** do professor
   — é o SFT clássico: "diga exatamente esta palavra". Sozinha, é o braço **ce**.
4. **Perda KL (sinal *soft*, dos logits).** Em cada posição: pega-se os **top-50 ids** do professor, aplica-se
   `softmax(logit_professor / T)` **renormalizado só sobre esses 50** (vira o alvo "suave"); do lado do aluno,
   `log_softmax` **nos mesmos 50 ids**; a KL aproxima uma distribuição da outra. É o braço **kl** — ensina não só
   "a palavra certa", mas *quais alternativas o professor considerou e com quanta confiança*.
5. **Combinação.** O braço **combinado** soma os dois sinais: `perda = α·CE + (1−α)·T²·KL` (com `α = 0,5`). O fator
   **T²** reescala o gradiente da KL para que ele não fique fraco quando `T > 1` — mantém CE e KL na mesma ordem de
   grandeza.
6. **Backward + update.** Backprop da perda combinada, acumulando gradiente por 16 micro-batches antes de cada
   passo do otimizador. **Fine-tuning completo** (o aluno é pequeno), `bfloat16` + *gradient checkpointing* para
   caber na L4.

A comparação decisiva cai diretamente desse laço: **o passo 4/5 (logits) entrega um aluno melhor que o passo 3
sozinho (só texto)?** Se sim — e foi o que ocorreu no 1.5B —, é prova de que o sinal *soft* dos logits transfere
algo **além do texto**. Note que **nada disso depende de RAG**: o laço é idêntico nos braços A e B; o RAG só muda
*qual texto/quais logits* o professor produziu lá atrás, na geração do dataset.

**Organização dos jobs:** um job gera o dataset (reconstruindo o índice RAG se faltar, para o braço B) e outro varre
a matriz **{0.5B,1.5B} × {ce,kl,combined} × {A,B} = 12 alunos**, despachando alunos em paralelo, um por GPU.

---

## 6. Resultados do núcleo — houve transferência?

Pipeline executado no cluster (2× GPU L4): geração do dataset, destilação dos 12 alunos e avaliação. Métricas no
benchmark held-out de 100 perguntas, **sem RAG na inferência** (testa o que ficou *nos pesos*). A referência é a
resposta do professor com RAG. RG = ROUGE-L; KR = key_recall.

### 6.1 Núcleo — destilação monolítica (braço A, sem RAG em lugar nenhum)

Estes 6 alunos são a **implementação padrão** exigida pelo enunciado: professor respondendo **da própria memória**,
aluno destilado por CE / KL / combinado. É a tabela que **responde a pergunta central** — sem nenhuma mistura de RAG.

| Modelo | geral RG | geral KR | DOM RG | DOM KR | doc RG | doc KR |
|---|---|---|---|---|---|---|
| base 0.5B | 0,227 | 0,380 | 0,296 | 0,505 | 0,157 | 0,249 |
| base 1.5B | 0,185 | 0,366 | 0,230 | 0,453 | 0,141 | 0,276 |
| d_0.5b · ce | 0,220 | 0,550 | 0,244 | 0,642 | 0,196 | 0,453 |
| d_0.5b · kl | 0,174 | 0,577 | 0,184 | 0,660 | 0,164 | 0,490 |
| d_0.5b · comb | 0,194 | 0,563 | 0,214 | 0,654 | 0,174 | 0,469 |
| **d_1.5b · ce** | **0,244** | **0,617** | 0,293 | 0,661 | 0,196 | 0,570 |
| d_1.5b · kl | 0,201 | 0,576 | 0,232 | 0,664 | 0,170 | 0,484 |
| d_1.5b · comb | 0,208 | 0,582 | 0,241 | 0,619 | 0,176 | 0,544 |

**Conclusões do núcleo:**
1. **Sim, houve transferência — inequívoca.** Os 6 alunos superam ambas as bases no key_recall (0,37–0,38 →
   0,55–0,62). O aluno-base quase não conhece os fatos; o destilado os internaliza nos pesos. Melhor do núcleo:
   **1.5B · combinado de sinal / ce**, KR 0,617 (**+69%** sobre a base 1.5B).
2. **O sinal está no key_recall, não no ROUGE-L.** O RG quase não se move (o aluno **reformula** com outras palavras);
   o ganho de conhecimento aparece na **presença dos fatos** (KR). Por isso o KR é a métrica-guia.
3. **Entre CE / KL / combinado, o núcleo empata.** Com o professor respondendo de memória, os três sinais rendem
   praticamente o mesmo (no 1.5B, o CE puro até lidera com 0,617). A vantagem **limpa** dos *soft labels* só emerge
   quando o professor recebe os fatos certos — ver o extra 6.2.
4. **A destilação destrava o aluno maior:** a base 1.5B era *pior* que a 0.5B (RG 0,185 vs 0,227), mas, destilada,
   passa a liderar — a capacidade extra só se realiza com o sinal do professor.

### 6.2 🧩 Extra — professor aterrado por RAG (braço B)

*(Vai além do enunciado — mistura destilação + recuperação.)* Aqui o professor responde **com o contexto recuperado
do índice DOM-PI** (§3.7) antes de gerar os rótulos. O aluno continua **sem RAG** na inferência — só muda a qualidade
do professor que o ensinou.

| Modelo | geral RG | geral KR | DOM RG | DOM KR | doc RG | doc KR |
|---|---|---|---|---|---|---|
| d_0.5b · B · ce | 0,203 | 0,667 | 0,249 | 0,613 | 0,157 | 0,723 |
| d_0.5b · B · kl | 0,224 | 0,598 | 0,267 | 0,630 | 0,181 | 0,564 |
| d_0.5b · B · comb | 0,243 | 0,625 | 0,276 | 0,611 | 0,211 | 0,639 |
| d_1.5b · B · ce | 0,223 | 0,647 | 0,277 | 0,641 | 0,170 | 0,652 |
| d_1.5b · B · kl | 0,350 | 0,689 | 0,380 | 0,654 | 0,320 | 0,725 |
| **🏆 d_1.5b · B · comb** | **0,363** | **0,717** | **0,429** | 0,659 | 0,297 | **0,776** |

**O que o extra acrescenta:**
1. **Aterrar o professor eleva ainda mais o recall factual:** o melhor de todos os braços é o **1.5B · B · combinado**
   (KR 0,717) — **+96%** sobre a base e **+16%** sobre o melhor do núcleo (6.1). Mais nítido em docentes (KR 0,776).
2. **Agora a vantagem dos *soft labels* é limpa:** no 1.5B-B, combinado (RG 0,363) ≥ kl (0,350) ≫ ce (0,223) — os
   **logits** transferem algo **além do texto**, mas isso só se manifesta quando o professor tem os **fatos certos**.
3. **Ponte Q4↔Q5:** aterrar o professor (recuperação) melhora o que o aluno **internaliza** — o elo direto com a Q5.

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

### ⭐ Atualização v2 — benchmark factual + especialistas por tópico

**Problema do v1:** 71% das referências eram abstenções ("Não consta") — a pergunta vinha de uma passagem, mas a
referência usava contexto RAG sobre todo o índice; com recall ~42%, o documento-fonte muitas vezes não era recuperado
e o professor se abstinha. **Correção (v2):** usar a **passagem-fonte como contexto-ouro** da referência → **0 "Não
consta"** (99% factual). E **seccionar a destilação por tópico** (DOM-PI × docentesDC): um especialista por domínio,
avaliado no seu próprio benchmark. Scripts: `--gold-context` em `gerar_dataset_destilacao.py`/`gerar_benchmark.py`;
orquestrador `run_q4_topicos.sbatch`. Artefatos em `dados_v2/`, `modelos/esp_*`, `resultados/avaliacao_{dompi,docentes}.json`.

| Tópico | Modelo | ROUGE-L | key_recall | Δ KR |
|---|---|---|---|---|
| DOM-PI | base 0.5B → esp 0.5B | 0,373 → 0,520 | 0,515 → 0,526 | +2% |
| DOM-PI | base 1.5B → **esp 1.5B** | 0,231 → 0,380 | 0,488 → **0,537** | +10% |
| docentesDC | base 0.5B → esp 0.5B | 0,218 → 0,308 | 0,352 → 0,412 | +17% |
| docentesDC | base 1.5B → **esp 1.5B** | 0,171 → 0,221 | 0,328 → **0,404** | +23% |

**Leitura:** com referências factuais, o ganho aparece **nas duas métricas** (não só ROUGE-L, mas key_recall até
+23%) — a destilação transfere **fatos reais**, não a disciplina de abstenção que dominava o headline do v1. Ganho
maior em docentesDC (domínio menos conhecido pela base). Gráfico: `resultados/figuras/topicos_especialistas.png`.

### Visualizações (`resultados/figuras/`)
Os gráficos principais focam no **núcleo monolítico** (braço A, sem RAG) para não poluir; as misturas de técnica
(braço B, cross-família, etc.) ficam em figuras/seções à parte. Gerados por `scripts/graficos_destilacao.py` e
`scripts/comparativo_por_questao.py`:

**Núcleo (monolítico):**
- `barras_keyrecall_nucleo.png` — key_recall das 8 configs do núcleo (2 bases + 6 do braço A), base como referência;
- `heatmap_keyrecall.png` — key_recall pergunta × modelo, **só o núcleo** (8 linhas), com faixa abstenção/fato;
- `box_por_metodo.png` — distribuição de key_recall por método (ce/kl/combinado) no braço A;
- `abstencao_vs_fato.png` — onde mora o ganho: KR médio em abstenção vs fato real (núcleo);
- `antes_depois_dominio.png` — base → melhor do núcleo (1.5B·ce) por domínio (+46% DOM-PI, +107% docentesDC);
- `compressao_vs_keyrecall.png` — custo-benefício: 1.5B é 9× menor e 0.5B 28× menor que o professor 14B.

**🧩 Extras (misturas de técnica):**
- `barras_keyrecall_extra_B.png` — o braço B (professor aterrado por RAG) e seu ganho adicional;
- `topicos_especialistas.png` — especialistas por tópico (v2, contexto-ouro); `delta_base_vs_melhor.png` — ganho por
  pergunta do campeão (extra) sobre a base.

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
| **Este trabalho** | Qwen | white-box (top-50 logits) | **9× (1.5B) / 28× (0.5B)** | key_recall **+69%** (núcleo) / **+96%** (extra RAG) vs base |

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

- **Núcleo (exigência) vs extra:** o núcleo é a **destilação monolítica** (braço A, sem RAG) — 6 alunos
  {tamanho}×{sinal}; aterrar o professor por RAG (braço B) e as demais extensões (§§7–12) são **🧩 extras**.
- **Por que 6+6 alunos:** desenho fatorial {tamanho} × {sinal} (núcleo) × {grounding A/B} (o eixo B é o extra), para
  isolar cada efeito mudando **um fator por vez**.
- **Por que esses modelos:** professor forte e **da mesma família** (habilita logits); alunos base pequenos (0.5B
  dá o contraste máximo); o modelo fraco da Q1 não serve de professor (transferiria erros).
- **Resultado-âncora (núcleo):** houve transferência inequívoca — melhor monolítico **1.5B · ce, +69%** no KR; entre
  CE/KL/combinado o núcleo praticamente empata.
- **O que o extra acrescenta:** aterrar o professor por RAG leva ao campeão global **1.5B · B · combinado, +96%** e
  torna **limpa** a vantagem dos logits (soft > texto).
- **Honestidade:** as métricas coletadas são **ROUGE-L e key_recall** (não PPL); toda pergunta passa por todo modelo,
  greedy e sem RAG. No v1, o ganho foi sobretudo **confiabilidade** (parar de alucinar); o v2 factual e a Copa 2026
  mostram **recordação factual real**; e o RAG na inferência (Q5) segue necessário para precisão.

## Referência metodológica
- Sebastian Raschka — *Build a Large Language Model (From Scratch)*: logits e softmax na geração de texto;
  cross-entropy e perplexidade como objetivo e métrica de treino; convenção de alinhamento de rótulos (base da
  correção do bug de duplo deslocamento reaproveitada aqui na máscara de perda do aluno).
