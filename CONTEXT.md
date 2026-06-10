# CONTEXTO — Panorama Atual do Pipeline DOM-PI

Arquitetura de dados e estado atual do pipeline de construção do **corpus do Diário Oficial
dos Municípios do Piauí (DOM-PI)** para **LLM/RAG**. Documento de orientação técnica; o
relatório de construção do corpus está em [`RELATORIO_CORPUS_DOM-PI.md`](RELATORIO_CORPUS_DOM-PI.md);
as demandas de limpeza do dataset publicado estão em [`CONTEXT_2.md`](CONTEXT_2.md).

> Histórico: versões anteriores deste pipeline usavam 3 scripts atômicos
> (`pipeline.py` → `download_pdfs.py` → `processar_pdfs.py`, extração CPU/PyMuPDF, só
> Carnaubais). Essa arquitetura foi **substituída** pela descrita abaixo (orquestração
> única + extração GPU + data lake colunar, 12 territórios).

---

## 1. Estado atual do corpus

- **~67,7 mil documentos** · **~195 milhões de tokens** · **12 dos 13 Territórios de Desenvolvimento** · publicações de **2025**.
- Publicado no HuggingFace: [`gutoportelaa/dom-pi-corpus-2025`](https://huggingface.co/datasets/gutoportelaa/dom-pi-corpus-2025) (CC-BY-4.0).
- **Fora da cobertura:** Teresina e Parnaíba (publicam em diários próprios, fora do DOM-PI dos Municípios).
- Custo aproximado da extração: ~75,6 GPU-horas (L4, cluster SLURM).

---

## 2. Natureza dos dados e problema central

### 2.1 Edição compartilhada
O DOM-PI publica **edições consolidadas**: um mesmo PDF pode conter publicações de múltiplos
municípios, entidades (Prefeitura, Câmara) e categorias de ato. Logo:
- a mesma URL de PDF aparece em dezenas de linhas do portal, para cidades diferentes;
- baixar/treinar com o mesmo conteúdo repetido degrada o corpus ("poisoning").

A resposta é **deduplicação em camadas**, separando *aparição no diário* de *arquivo físico*
de *conteúdo textual*.

### 2.2 PDFs escaneados vs. nativos
Boa parte dos PDFs são **escaneados** (imagem), não texto nativo. A extração precisa, então,
de **OCR** — e o resultado carrega ruído (cabeçalhos repetidos, assinaturas, tabelas achatadas).
Daí o roteamento por tipo de página e a etapa de limpeza/flag de qualidade.

---

## 3. Arquitetura end-to-end

```
scraping  →  download  →  reconstrução  →  EXTRAÇÃO (GPU/SLURM)  →  DATA LAKE (CPU local)  →  dataset
scraper_   scrape_/      reconstruir_     orquestrador_           datalake/                  HF Hub
isolado.py download_     coleta.py        extracao.py             extraído→limpo→corpus
           demais.sh                      (PyMuPDF→Paddle/Docling)
```

| Etapa | Onde | Módulo / artefato |
|---|---|---|
| 1. Scraping de metadados | local | `scraper_isolado.py` → `dados/scraping_results/*.json` |
| 2. Download de PDFs | local | `scrape_demais.sh` / `download_demais.sh` → `territorios/<slug>/` |
| 3. Reconstrução da estrutura | local | `reconstruir_coleta.py` (manifesto → árvore território/município) |
| 4. Extração (pesada) | cluster GPU | `orquestrador_extracao.py` via `run_extracao.sbatch` |
| 5. Estruturação do lake | local CPU | pacote `src/dompi_scraper/datalake/` |
| 6. Publicação | local | export Parquet + shards `.jsonl.zst` → HuggingFace |

> Fronteira lab↔local desacoplada: o cluster entrega NDJSON por território; o data lake
> ingere localmente, sem reextrair.

---

## 4. Extração (etapa 4) — roteamento por tipo de página

`orquestrador_extracao.py` orquestra os workers (`engine_worker.py`) com triagem + roteamento:

1. **Triagem PyMuPDF** — lê a página e decide a rota pela densidade de texto/estrutura.
2. **Rota PaddleOCR-CUDA** — páginas comuns/escaneadas (a maior parte do volume).
3. **Rota Docling-CUDA** — páginas fiscais/com tabelas (RGF, RREO, planilhas de licitação),
   onde preservar a estrutura `| coluna | coluna |` importa para o RAG.
4. Saída por território: NDJSON (1 linha/documento) com texto + proveniência, no schema
   `_CORPUS_SCHEMA` (contrato em `orquestrador_extracao.py`).

Operação no cluster: SLURM (`--gres=gpu:l4:1 --mem=28G`), com auto-resubmit encadeado.
**Nunca** executar a extração via `uv run` — usar o interpretador do venv (`./.venv/bin/python`).

---

## 5. Limpeza e flags de qualidade (`limpeza_textos.py`)

Transforma a camada **extraído → limpo**:
- `clean_text()` remove ruído **preservando tabelas Markdown**;
- re-hash do conteúdo pós-limpeza (dedup de itens que ficam idênticos após limpar);
- **flags por severidade** — `assinaturas_detectadas` é informativo; `needs_human_review`
  marca só alta severidade (`tabela_achatada_detectada`, `alto_indice_ruido_ocr`);
- métricas `n_chars`, `n_tokens`.

> Triagem histórica (Carnaubais, 7.565 docs) mostrou ~95% `needs_human_review` quando a flag
> marcava qualquer assinatura; a regra por severidade reduz isso à fração de alta severidade.
> O detalhamento do plano VLM/OCR que originou esse roteamento virou as demandas em `CONTEXT_2.md`.

---

## 6. Data lake (etapa 5) — camadas extraído / limpo / corpus

```
datalake/
  extraido/  territorio=<slug>/ano=<AAAA>/part-*.parquet   # 1 linha/doc + proveniência
  limpo/     territorio=<slug>/ano=<AAAA>/part-*.parquet   # texto limpo, re-hash, dedup, flags
  corpus/    corpus_llm/ part-*.parquet (+ shards .jsonl.zst)  # pronto p/ treino
  _catalog/  manifest.parquet · dedup_global.parquet
```

- Stack: **DuckDB + Polars + Parquet/zstd**. Particiona por `territorio` + `ano` (não por mês:
  datas incompletas degenerariam as partições). `tipo_ato` é coluna, não partição.
- CLI (CPU-leve, local):
  ```bash
  python -m dompi_scraper.datalake.ingest_extraido --territorio <slug>   # ou --all
  python -m dompi_scraper.datalake.build_limpo      --territorio <slug>   # ou --all
  python -m dompi_scraper.datalake.build_corpus
  python -m dompi_scraper.datalake.query "SELECT territorio, count(*) FROM corpus GROUP BY 1"
  ```
- Correções pós-extração (sobre a camada extraído, com rebuild de limpo+corpus):
  `corrigir_datas.py` (cronologia ancorada na data da **edição**) e `corrigir_municipios.py`
  (canoniza o município contra a lista oficial em `to-do_territorios.txt`).

---

## 7. Esquema do corpus publicado

Cada linha é **um ato/página publicado**:

| Coluna | Descrição |
|---|---|
| `id` | MD5 do conteúdo normalizado (chave de dedup) |
| `territorio` | Território de Desenvolvimento (slug) |
| `municipio` | nome oficial canonizado (176 valores; `DESCONHECIDO` quando irrecuperável) |
| `tipo_ato` | Portaria, Decreto, Lei, Edital, Licitação, Contrato, LRF, Ata… (regex) |
| `ano` | ano de publicação (2025) |
| `data_publicacao` | `DD/MM/AAAA` (91%) ou `2025` (sem data precisa) |
| `n_tokens` | estimativa (~`n_chars/4`) |
| `texto` | texto limpo do documento |

---

## 8. Deduplicação em camadas

| Camada | Chave | Onde |
|---|---|---|
| L1 — aparição→arquivo | `md5(url)` | scraping/download (evita re-download de PDF compartilhado) |
| L2 — conteúdo pós-extração | `md5(texto)` | ingest extraído |
| L3 — conteúdo pós-limpeza | `md5(texto_limpo)` | build limpo (re-hash) |
| L4 — global cross-território | `id_limpo` | `_catalog/dedup_global.parquet` |

O hash de conteúdo normaliza (minúsculas, espaços/quebras colapsados, controles Unicode removidos)
e **não** inclui URL/metadados. Limitação: é **dedup exato** — quase-duplicatas persistem
(ver `CONTEXT_2.md`).

---

## 9. Módulos do pacote `src/dompi_scraper/`

| Módulo | Papel |
|---|---|
| `orquestrador_extracao.py` | orquestra a extração GPU (triagem + roteamento + schema) |
| `engine_worker.py` | worker de extração por documento |
| `extrair_territorio.py` | caminho de extração por território (legado/alternativo) |
| `limpeza_textos.py` | `clean_text` + flags de qualidade (extraído→limpo) |
| `shared_utils.py` | slugify, hash, `classify_act`, utilitários comuns |
| `territorios_pi.py` | registro dos 13 Territórios e municípios |
| `reconstruir_coleta.py` | manifesto de download → árvore território/município |
| `datalake/` | camadas do lake (ver §6) |
| `selenium/`, `../vector_db/` | apoio de coleta e ingestão vetorial (RAG) |
