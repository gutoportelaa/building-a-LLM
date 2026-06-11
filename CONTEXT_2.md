# CONTEXTO 2 — Demandas de Limpeza do Dataset (HuggingFace)

Backlog de limpeza/qualidade do corpus **publicado** em
[`gutoportelaa/dom-pi-corpus-2025`](https://huggingface.co/datasets/gutoportelaa/dom-pi-corpus-2025)
(~67,7 mil docs, ~195M tokens, 12 territórios, 2025). O panorama da arquitetura está em
[`CONTEXT.md`](CONTEXT.md). Este documento lista o que **ainda precisa ser feito** para elevar
a qualidade do dataset, priorizado por impacto.

> Histórico: este arquivo nasceu como o plano de triagem VLM/OCR (reconstruir ~7 mil docs
> "achatados" do Carnaubais via Marker/VLM). Aquela direção foi **realizada** — a extração
> hoje roteia páginas fiscais/tabelas para **Docling-CUDA** e o restante para **PaddleOCR-CUDA**
> (ver `CONTEXT.md` §4). O que segue é o backlog remanescente sobre o corpus já gerado.

---

## Estado de qualidade conhecido (limitações declaradas no dataset card)

| Dimensão | Situação atual |
|---|---|
| Cobertura | 12 de 13 Territórios; **sem Teresina e Parnaíba** |
| Município | canonizado para 176 valores oficiais; **~3,2% (~2.147 docs) `DESCONHECIDO`** |
| Data | **91% com `DD/MM/AAAA`**; ~9% só com o ano (`2025`) |
| Dedup | só **exato** (pós-normalização); quase-duplicatas persistem |
| Ruído OCR | cabeçalhos/assinaturas residuais; ~50% com `needs_review` interno (não exposto) |
| Mega-documentos | edições/leis consolidadas muito longas (LOAs de centenas de páginas) |

---

## ✅ Resolvido (implementado no `build_corpus`)

### D-1 · Quase-duplicatas (near-dups) — FEITO
Módulo `datalake/dedup_aproximada.py`: **MinHash (128 perm) + LSH** sobre shingles de
5-gramas de palavras do texto normalizado, limiar **Jaccard 0,85**; componentes conexos →
clusters; elege 1 **canônico** por cluster (maior texto → menos flags → data mais antiga).
Resultado: **863 redundantes removidos** (1,1%) em 756+ clusters — duplicatas reais (LRF
curtos quase idênticos, pares de licitação/portaria). O split **`train`** mantém só
canônicos; o **`raw`** preserva todos com `cluster_id`/`is_near_dup` (reversível). Lever:
`--threshold 0.80` para mais recall. Catálogo em `datalake/_catalog/near_dup.parquet`.

### D-2 · Documentos longos — FEITO (com achado relevante)
Módulo `datalake/fatiar_megadocs.py`. **Medição contradisse a premissa inicial:** os docs
>32k tokens **não** são compilações de muitos atos — 76% são **documentos únicos** (orçamento,
planilha de licitação, uma LOA). As compilações reais estão em **8k–32k**. Decisão: fatiar
**todos os docs >8k** que sejam compilações, usando fronteira = **título de ato** (PORTARIA/
DECRETO/LEI/… Nº em início de linha) — preciso e **seguro** (em tabela fiscal acha 0 fronteiras
→ mantém intacto, não corta tabela). Resultado: **1.106 docs fatiados → 10.645 atos**; 1.770
docs únicos longos mantidos intactos. Coluna **`tamanho_classe`** (normal/longo/mega) em todas
as linhas: normal 74.757 · longo 1.850 · mega 532. Tokens preservados (~193,7M).

> Produto: `corpus/corpus_llm` (train, 76.276 docs) + `corpus/corpus_raw` (raw, 77.139).
> Empacotamento reproduzível em `datalake/empacotar_hf.py` (gera `hf_corpus_dompi/` com
> configs `default`/`raw`). **Re-publicação no HF é passo manual** (não automatizado).

## Análise de qualidade textual (2026-06-10) — base p/ níveis de limpeza

Inspeção do `texto` do corpus revelou **três tipos de ruído distintos**, com tratamentos diferentes:

1. **Boilerplate** (≈50–56%): cabeçalho de diário (`Ano XXIII · Teresina (PI) - … · Edição V`),
   placeholder `-- image -->`, linhas de QR/autenticidade/URL, assinaturas com `____`, nº de página.
   **Removível por regex.** O `clean_text` atual NÃO pega o cabeçalho (a regra exige `«`, mas o
   diário usa `·`/`•`). → **limpeza v2** corrige isso (boilerplate strip + dedup de linha repetida).
2. **Corrupção de OCR no nível da palavra** (ex.: "Homo·oaacão", "ucuatórlo", "pUblleo",
   "lndk:lldorN"): **não** conserta por regex. Parcialmente detectável por `real_word_ratio`
   (dicionário PT via `wordfreq`).
3. **Tabela fiscal achatada** (RGF/RREO/orçamento virados em sopa de números/códigos):
   lexicamente "ok" (palavras reais), semanticamente inútil. **Não detectável por dicionário** —
   só por sinal **estrutural** (densidade numérica). É problema de **re-extração** (Docling/VLM,
   ver D-5/D-6), não de limpeza.

**Tiers de qualidade** (amostra 9k, métrica `real_word_ratio` × densidade numérica):

| Tier | Critério | % | Uso |
|---|---|---|---|
| **A · prosa limpa** | real_word ≥ 0,88 e numérico < 0,15 | **~52%** | SFT / instruction |
| **B · média** | demais legíveis | ~35% | pré-treino |
| **C · tabela achatada/ruim** | numérico ≥ 0,35 ou real_word < 0,78 | **~13%** | excluir do treino de prosa; re-extrair |

O tier C concentra-se em **LRF (43%)**, Lei (14%), Decreto (11%) — os atos fiscais. Portaria/
Contrato/Licitação são limpos (2–4% C). O conteúdo que interessa para Q&A (vencedores de
licitação, valores) está nos tiers A/B (ex.: "Participante Vencedor: ANTARES … LTDA").

**Veredito MLOps:** ~87% (A+B) é texto aproveitável; 193M tokens. **Suficiente para EXPERIMENTAR
pré-treino** (escala Raschka) já — basta limpeza v2 + coluna `quality_tier`. Para **SFT/instruction**,
usar **Tier A**. Tabela achatada (C) é re-extração, fora do escopo de limpeza.

**Plano de níveis de limpeza → configs HF:** `extraido` (bruto, própria extração) · `pretrain`
(limpeza v2, todos os tiers + coluna `quality_tier`) · `pretrain-curado` (Tier A+B) · `instruction`
(futuro: Q&A gerado por LLM sobre Tier A — **não** templar de metadados, pois "proprietários de
licitação" etc. estão só no texto). Implementação: `limpeza v2` + tiering no `build_limpo`/
`build_corpus`; exportador multi-config no `empacotar_hf`.

## Demandas priorizadas (pendentes)

### D-3 · Município `DESCONHECIDO` (~3,2%) — MÉDIA
**Problema:** ~2.147 docs não foram resolvidos contra a lista oficial por território.
**Abordagem:** **fuzzy match** (Levenshtein/`rapidfuzz`) do nome OCR contra a lista oficial de
`to-do_territorios.txt`, com limiar conservador; usar também o nome do arquivo (loghash) e o campo
`municipio` do scraping. Meta: reduzir `DESCONHECIDO` para o irredutível (PDFs realmente multi-município).

### D-4 · Datas incompletas (~9% só com ano) — MÉDIA
**Problema:** ~9% dos docs têm apenas `2025` em `data_publicacao`.
**Abordagem:** completar o mapa **edição→data** a partir dos metadados de scraping (e dos logs do lab),
fechando os 1:1 faltantes; reprocessar via `corrigir_datas.py`. Habilita particionar/filtrar por mês.

### D-5 · Ruído residual de OCR / `needs_review` — MÉDIA
**Problema:** ~50% dos docs têm `needs_review` interno (não exposto no HF); restam cabeçalhos
repetidos e assinaturas no `texto`.
**Abordagem:** decidir a política de exposição — (a) **expor** colunas de qualidade
(`needs_review`, `assinaturas_detectadas`, score OCR) para o consumidor filtrar; ou (b) oferecer um
**split/config "curado"** já filtrado. Adicionar limpeza de cabeçalho/rodapé repetido por edição.

### D-6 · Correção de raiz na extração (município/data da fonte, não do OCR) — MÉDIA/ESTRUTURAL
**Problema:** `municipio`/`data` foram capturados do **conteúdo OCR**, exigindo correção posterior.
**Abordagem:** alterar `orquestrador_extracao.py` para tomar `municipio`/`data` do **caminho/manifesto**
(árvore território/município de `reconstruir_coleta.py` + edição→data), eliminando a origem do erro
em futuras extrações. Torna D-3/D-4 desnecessárias daqui para frente.

### D-7 · Cobertura: Teresina e Parnaíba — BAIXA (esforço alto)
**Problema:** ambas publicam fora do DOM-PI dos Municípios. Parnaíba (`diarios_parnaiba/`) é um
**SPA Quasar ("DOMe")** — o download trouxe 217 stubs HTML idênticos de 603 bytes, não PDFs.
**Abordagem:** descobrir a **API** do portal (XHR/JSON) ou usar **navegador headless** para obter os
PDFs reais; só então passar pela pipeline padrão. Os stubs atuais são descartáveis.

---

## Sequência sugerida

1. ✅ **D-1 + D-2** — feitos no `build_corpus` (train/raw). Falta **re-publicar** o HF
   (`empacotar_hf` já gerou `hf_corpus_dompi/`; upload é manual).
2. **D-3 + D-4** (fuzzy município + completar datas) → re-publicar.
3. **D-5** decidir exposição de flags / split curado.
4. **D-6** corrigir a raiz na extração (beneficia novas coletas).
5. **D-7** Parnaíba/Teresina quando houver esforço para o SPA/diário próprio.

> Todos os passos de D-1 a D-5 são **CPU-leves e locais** (Polars/DuckDB sobre texto), sem GPU.
> Cada re-publicação regenera `hf_corpus_dompi/` (Parquet + shards) a partir de `datalake/corpus`.
