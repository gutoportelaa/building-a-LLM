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

## Demandas priorizadas

### D-1 · Quase-duplicatas (near-dups) — ALTA
**Problema:** a dedup é por hash de conteúdo **exato**; variações de OCR, rodapés ou paginação
geram registros quase idênticos que escapam.
**Impacto:** repetição de conteúdo no treino ("poisoning") e inflação de contagens por município/tipo.
**Abordagem:** dedup aproximada por **MinHash/LSH** ou **SimHash** sobre `texto` normalizado
(shingles de n-gramas), com limiar de similaridade calibrado; manter a 1ª ocorrência e registrar
o cluster em `_catalog/dedup_global.parquet`. Rodar como passo extra em `build_corpus`.

### D-2 · Mega-documentos — ALTA
**Problema:** alguns registros são edições/leis consolidadas enormes (LOA, planos plurianuais),
muito acima da mediana de tokens.
**Impacto:** distorcem estatísticas, dominam batches de treino e estouram janelas de contexto.
**Abordagem:** detectar por `n_tokens` (corte por percentil), **sinalizar** com coluna de tamanho
e/ou **fatiar** em unidades de ato preservando tabelas; decidir política (expor flag vs. dividir).

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

1. **D-1 + D-2** sobre `datalake/limpo` → reconstruir `corpus` → **re-publicar** o HF (maior ganho de qualidade).
2. **D-3 + D-4** (fuzzy município + completar datas) → re-publicar.
3. **D-5** decidir exposição de flags / split curado.
4. **D-6** corrigir a raiz na extração (beneficia novas coletas).
5. **D-7** Parnaíba/Teresina quando houver esforço para o SPA/diário próprio.

> Todos os passos de D-1 a D-5 são **CPU-leves e locais** (Polars/DuckDB sobre texto), sem GPU.
> Cada re-publicação regenera `hf_corpus_dompi/` (Parquet + shards) a partir de `datalake/corpus`.
