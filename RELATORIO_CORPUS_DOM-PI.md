# Corpus DOM-PI — Relatório de Construção do Dataset

> Documento de handoff. Descreve **todo o trabalho** da sessão: ambientação da
> infraestrutura, scraping, download, extração, estruturação do data lake e
> correção de cronologia — até o dataset pronto para versionar no HuggingFace.
> Serve também como ponto de partida para a próxima sessão (limpeza, tratamentos,
> versionamento, RAG).

**Data:** junho/2026 · **Status:** ✅ dataset consolidado e corrigido, pronto para HF.

---

## 1. Sumário executivo

| Métrica | Valor |
|---|---|
| Territórios de Desenvolvimento (TDs) processados | **12** (de 13; Teresina e Parnaíba não existem na fonte) |
| PDFs varridos / extraídos (camada *extraído*) | **77.337** |
| **Documentos únicos** (após dedup, camada *corpus*) | **67.687** |
| **Tokens estimados** | **~195,3 milhões** |
| Tempo total de GPU em extração | **~75,6 GPU·hora** (L4) |
| Cronologia | **100% ano 2025** corrigido (P-03 eliminado); 91% com data DD/MM/AAAA |
| Saída | `datalake/corpus/corpus_llm/` — 14 shards `.jsonl.zst` + Parquet por ano (360 MB) |

O corpus é o texto integral das publicações de 2025 do **Diário Oficial dos
Municípios do Piauí (DOM-PI)**, organizado por Território de Desenvolvimento e
por município, limpo e deduplicado.

---

## 2. Infraestrutura

Dois ambientes, com papéis separados de propósito:

| Ambiente | Papel | Detalhes |
|---|---|---|
| **Cluster lab (SLURM)** `aluno_matheus@10.94.80.10` | Scraping, download e **extração com GPU** | `gpunode01` = 2× NVIDIA L4; `gpunode02` = 1× L4; 16 CPU e ~62,9 GB RAM por nó. Download roda no *login node* (tem internet). |
| **WSL local** (`~/Documents/building-a-LLM`) | Reconstrução, **estruturação do data lake** (CPU-leve) e correção | 24 GB RAM (cap via `.wslconfig`); ingestão roda sob `systemd-run --scope -p MemoryMax=…` + `monitor_os.sh` para nunca travar o host. |

**Ambientação realizada nesta sessão:**
- **Liberação de 662 GB de disco local:** `dados/chroma_db` ocupava 663 GB, mas era **corrupção** (um `link_lists.bin` HNSW esparso/quebrado de um índice RAG antigo), não dados. O índice real (30.354 vetores) eram <700 MB. Backup verificado em `dados/chroma_db_backup_2026-06-05/` e removido só o arquivo corrompido. Disco local: 89% → 20%.
- **GPU/SLURM:** NVML quebrado no `gpunode01` (contar GPU por env do SLURM, não `nvidia-smi`); `--mem=28G` por job (caber 2 jobs no nó de 62,9 GB → paralelismo real); `--time=08:00:00 --gres=gpu:l4:1` para habilitar *backfill*; `--signal=B:TERM@120` + checkpoint para sobreviver ao walltime.

---

## 3. Arquitetura do data lake

Camadas nomeadas pela função (sem jargão *medallion*):

```
datalake/
  extraido/  territorio=<slug>/ano=<AAAA>/*.parquet   (1 linha/doc + proveniência)
  limpo/     territorio=<slug>/ano=<AAAA>/*.parquet   (clean_text + dedup L3 + flags)
  corpus/    corpus_llm/{ano=<AAAA>/*.parquet, shards/*.jsonl.zst}
  _catalog/  manifest.parquet + dedup_global.parquet
```

**Tecnologias:** DuckDB (SQL/COPY particionado), Polars (transformações), PyArrow (ponte), Parquet+zstd, shards `.jsonl.zst` (formato de treino).

**Módulos / CLI** (rodar `./.venv/bin/python -m dompi_scraper.datalake.<m>`):
| Módulo | Função |
|---|---|
| `ingest_extraido --territorio <slug> --source <corpus.jsonl>` | NDJSON da extração → camada **extraído** |
| `build_limpo --territorio <slug> \| --all` | extraído → **limpo** (limpeza, re-hash, dedup, flags) |
| `build_corpus` | limpo → **corpus** (Parquet + shards `.jsonl.zst`) |
| `query "<sql>"` | consultas DuckDB (views `bronze/silver/gold`) |
| `corrigir_datas --map … --log …` | corrige cronologia (P-03) — ver §8 |
| `reconstruir_coleta --manifest … --territorio <slug>` | coleta *flat-hash* → árvore por município (recupera município+data no nome) |

**Pipeline do scraper** (raiz): `scraper_isolado.py` (+`pipeline.py`, registry `territorios_pi.py`) → `download_pdfs.py` → `reconstruir_coleta` → extração → ingestão.

---

## 4. Os 12 territórios

> "Scraping?" = precisou raspar metadados do DOM-PI nesta sessão.
> *extraídos* = docs antes do dedup cross-território; *únicos* = após dedup (atribuição ao 1º TD processado).

| Território (TD) | Scraping? | PDFs | extraídos | únicos (limpo) | GPU·h | needs_review |
|---|---|---|---|---|---|---|
| cocais (TD2) | ✅ sim | 22.222 | 12.188 | 11.475 | 11,1 | 48% |
| mangabeiras (TD12) | ✅ sim | 19.943 | 10.889 | 9.720 | 10,7 | 50% |
| vale_do_rio_guaribas (TD6) | ✅ sim | 15.059 | 8.715 | 6.897 | 9,0 | ~50% |
| serra_da_capivara (TD9) | ✅ sim | 15.795 | 8.747 | 7.271 | 7,3 | 47% |
| chapada_vale_do_rio_itaim (TD7) | ✅ sim | 13.459 | 7.496 | 7.268 | 6,0 | 41% |
| vale_do_caninde (TD8) | ✅ sim | 6.649 | 4.252 | 3.403 | 3,6 | 55% |
| tabuleiros_alto_parnaiba (TD11) | ❌ pré-existia | 13.293 | 7.115 | 5.990 | 8,8 | ~45% |
| carnaubais (TD3) | ❌ pré-existia | 12.370 | 7.380 | 7.380 | 6,7 | 47% |
| vale_dos_rios_piaui_e_itaueiras (TD10) | ❌ pré-existia | 9.800 | 6.166 | 4.664 | 6,1 | 60% |
| planice_litoran (TD1) | ❌ pré-existia | 4.386 | 2.723 | 2.301 | 2,9 | 55% |
| entre_rios (TD4) | ❌ pré-existia | 963 | 1.066 | 938 | 2,3 | 54% |
| vale_do_sambito (TD5) | ❌ pré-existia | 555 | 600 | 380 | 1,0 | 57% |
| **TOTAL** | | **~146.494** | **77.337** | **67.687** | **75,6** | ~50% |

**Motores de extração** (roteamento por triagem PyMuPDF): `docling-cuda` 39.024 docs (méd. 6,2 s — fiscais/tabelas), `paddle-cuda` 38.285 docs (méd. 0,76 s — comum/escaneado), `pymupdf-fallback` 28.

---

## 5. Etapas aplicadas (cronologia do trabalho)

1. **Ambientação** — liberação de 662 GB (chroma_db corrompido), ajustes de SLURM/GPU e do cap de RAM local.
2. **Territórios pré-existentes (6)** — `tabuleiros, planice_litoran, entre_rios, vale_do_sambito, vale_dos_rios` (PDFs já depositados pela equipe) e `carnaubais` (baixado em sessão legada, formato *flat-hash*). Cada um: transferência ao lab → extração → ingestão. O carnaubais exigiu **reconstrução** (`reconstruir_coleta`) porque os nomes eram hash MD5 sem município/data.
3. **Scraping dos 6 territórios sem PDFs** (ver §6) — `scraper_isolado.py --territorio <slug>` com registry `territorios_pi.py` (120 municípios validados ao vivo contra o formulário do DOM-PI). ~4 h 33 min, **93.127 URLs** únicas coletadas (`dados/scraping_results/scraping_<slug>_2025_deduplicados.json`).
4. **Download** (no *login node* do lab) — `download_demais.sh` → `download_pdfs.py` por território → `territorios/<slug>/pdfs_arquivos/<md5>.pdf` + `download_manifest.json`. **~93,1 mil PDFs / ~146 GB**, com retry exponencial 3× + *throttle* 0,5 s; relatório de falhas por TD (`logs/download_<slug>_FALHAS.txt` — **0 falhas**). Download direto no lab evitou transferir 146 GB do WSL.
5. **Reconstrução** — `reconstruir_coleta` lê o `download_manifest.json` e recria `territorios/<slug>/pdfs/<município>/<nome_descritivo>.pdf` via *hardlink* (recupera município e a data embutida no nome).
6. **Extração** — `run_extracao.sbatch` no SLURM, **até 3 GPUs L4 em paralelo** (1 território/GPU), auto-encadeada (checkpoint a cada 25 docs + no SIGTERM; re-submissão automática no corte de walltime). **~75,6 GPU·hora** no total.
7. **Ingestão** — por território: *extraído → limpo → corpus*, com **dedup em 4 camadas** (pré-extração por texto nativo; pós-extração por `content_hash`; re-hash pós-limpeza `id_limpo`; cross-território). Rodada local sob cap de RAM.
8. **Correção de cronologia (P-03)** — ver §8.

---

## 6. Cidades que necessitaram de scraping (120 municípios, 6 TDs)

> Fonte: registry validado em `src/dompi_scraper/territorios_pi.py`. No DOM-PI a forma
> "do Piauí" aparece abreviada "do Pi". **Teresina e Parnaíba ficaram de fora** — publicam
> em diário próprio, não no Diário Oficial *dos Municípios* (AMP).

- **TD2 — Cocais (22):** Barras, Batalha, Brasileira, Campo Largo do Piauí, Domingos Mourão, Esperantina, Joaquim Pires, Joca Marques, Lagoa de São Francisco, Luzilândia, Madeiro, Matias Olímpio, Milton Brandão, Morro do Chapéu do Piauí, Nossa Senhora dos Remédios, Pedro II, Piracuruca, Piripiri, Porto, São João da Fronteira, São João do Arraial, São José do Divino.
- **TD6 — Vale do Rio Guaribas (23):** Alagoinha do Piauí, Alegrete do Piauí, Aroeiras do Itaim, Bocaina, Campo Grande do Piauí, Dom Expedito Lopes, Francisco Santos, Fronteiras, Geminiano, Itainópolis, Monsenhor Hipólito, Paquetá, Picos, Pio IX, Santana do Piauí, Santo Antônio de Lisboa, São João da Canabrava, São José do Piauí, São Julião, São Luís do Piauí, Sussuapara, Vera Mendes, Vila Nova do Piauí.
- **TD7 — Chapada Vale do Rio Itaim (16):** Acauã, Belém do Piauí, Betânia do Piauí, Caldeirão Grande do Piauí, Caridade do Piauí, Curral Novo do Piauí, Francisco Macedo, Jacobina do Piauí, Jaicós, Marcolândia, Massapê do Piauí, Padre Marcos, Patos do Piauí, Paulistana, Queimada Nova, Simões.
- **TD8 — Vale do Canindé (17):** Bela Vista do Piauí, Cajazeiras do Piauí, Campinas do Piauí, Colônia do Piauí, Conceição do Canindé, Floresta do Piauí, Isaías Coelho, Oeiras, Santa Cruz do Piauí, Santa Rosa do Piauí, Santo Inácio do Piauí, São Francisco de Assis do Piauí, São Francisco do Piauí, São João da Varjota, Simplício Mendes, Tanque do Piauí, Wall Ferraz.
- **TD9 — Serra da Capivara (18):** Anísio de Abreu, Bonfim do Piauí, Campo Alegre do Fidalgo, Capitão Gervásio Oliveira, Caracol, Coronel José Dias, Dirceu Arcoverde, Dom Inocêncio, Fartura do Piauí, Guaribas, João Costa, Jurema, Lagoa do Barro do Piauí, São Braz do Piauí, São João do Piauí, São Lourenço do Piauí, São Raimundo Nonato, Várzea Branca.
- **TD12 — Chapada das Mangabeiras (24):** Alvorada do Gurguéia, Avelino Lopes, Barreiras do Piauí, Bom Jesus, Colônia do Gurguéia, Corrente, Cristalândia do Piauí, Cristino Castro, Currais, Curimatá, Eliseu Martins, Gilbués, Júlio Borges, Manoel Emídio, Monte Alegre do Piauí, Morro Cabeça no Tempo, Palmeira do Piauí, Parnaguá, Redenção do Gurguéia, Riacho Frio, Santa Filomena, Santa Luz, São Gonçalo do Gurguéia, Sebastião Barros.

Os demais TDs (`tabuleiros, planice_litoran, entre_rios, vale_do_sambito, vale_dos_rios, carnaubais`) **não precisaram de scraping** nesta sessão — já tinham PDFs.

---

## 7. Correção de cronologia (P-03)

**Problema:** a extração derivava o ano do **nome do arquivo** (frágil) → anos impossíveis (2001, 2054, 2099…). Antes da correção: 2.335 docs com ano-lixo + 1.858 `sem_ano`.

**Solução (`corrigir_datas.py`):** a data verdadeira é a **data da edição** do diário —
`edicao_url_meta → data_publicacao` é **1:1** (250 edições, todas 2025); o planície já
traz `DD-MM-AAAA_` no próprio nome. Ligou-se `id_publicacao` (hash do conteúdo) ↔ arquivo
pelos **logs do SLURM**, e arquivo → data pelo mapa (`dados/datas_map.json`: 105 mil
nomes exatos + 250 edições + interpolação para edições fora da amostra). Como a coleta é
**100% de 2025** (`scraper --ano 2025`), `ano=2025` para todos.

**Resultado:** ano **100% 2025** (lixo e sem_ano → 0); **91% (61.605 docs) com data precisa DD/MM/AAAA** (251 datas distintas). Totais do corpus inalterados.

---

## 8. Qualidade e ressalvas (para a próxima sessão)

1. **`needs_review` ~50%** (P-08) — flag por severidade (`tabela_achatada`, `alto_ruido_ocr`). Auditar o limiar antes de usar como filtro.
2. **Mega-documentos:** alguns docs têm **até 7 M caracteres** (LOAs / edições consolidadas) — outliers que dominaram o tempo de GPU. Candidatos a **filtrar ou segmentar**.
3. **Dedup é exato** (hash de conteúdo normalizado) — *near-duplicates* (1 char de diferença, ruído de OCR) sobrevivem. Camada fuzzy (MinHash/embeddings) seria o próximo nível.
4. **Atribuição de dedup cross-território** é dependente da ordem de processamento (o doc compartilhado conta no 1º TD processado). O **total (67.687) é estável**; a contagem por TD pode variar num rebuild.
5. **Erros de extração:** ~0,03% (`extracao_vazia`, páginas em branco/scan ruim) — registrados em `extraidos/<slug>/eventos_extracao.ndjson` (no lab).

---

## 9. Localização dos artefatos

| O quê | Onde |
|---|---|
| **Corpus final (HF)** | `datalake/corpus/corpus_llm/shards/*.jsonl.zst` (14) + `datalake/corpus/corpus_llm/ano=2025/*.parquet` |
| Camadas intermediárias | `datalake/{extraido,limpo}/` + `datalake/_catalog/` |
| Metadados de scraping | `dados/scraping_results/scraping_<slug>_2025_deduplicados.json` |
| Mapa de datas / link logs | `dados/datas_map.json` · `staging_lab/loghash.tsv` |
| Backup do índice RAG antigo | `dados/chroma_db_backup_2026-06-05/` (676 MB, 30.354 vetores) |
| Scripts-chave | `scraper_isolado.py`, `src/dompi_scraper/territorios_pi.py`, `download_demais.sh`, `run_extracao.sbatch`, `src/dompi_scraper/datalake/{reconstruir_coleta,corrigir_datas}.py` |
| PDFs por território | `territorios/<slug>/pdfs/<município>/…` (local e lab) |

---

## 10. Próximos passos sugeridos

- [x] **Versionar no HuggingFace** — subir `corpus_llm/` (shards `.jsonl.zst` p/ treino; Parquet p/ `datasets`). Montar `dataset_card`/README, definir splits.
- [ ] **Conserto permanente da data (camada A)** — patchar `orquestrador_extracao.py` para gravar a data pela edição **na extração** (+ `arquivo_origem`/`edicao`/`pagina`), tornando o `loghash.tsv` desnecessário e habilitando partição por **mês** (os 91% de datas precisas já permitem).
- [x] **Renomear camadas** `bronze/silver/gold` → `extraído/limpo/corpus` no código e nos diretórios (refactor isolado; dados são regeneráveis).
- [ ] **Limpeza das extrações** — revisar `needs_review`, near-dup (MinHash), filtro/segmentação dos mega-docs.
- [ ] **RAG** — reconstruir o índice vetorial do corpus atual (o build antigo de 30 k chunks no backup está desatualizado).

---

*Comandos úteis:* `./.venv/bin/python -m dompi_scraper.datalake.query "SELECT territorio, count(*) FROM corpus GROUP BY 1 ORDER BY 2 DESC"` (views: `extraido`/`limpo`/`corpus`) · ingestão sempre sob `systemd-run --user --scope -p MemoryMax=14G …` · **nunca** `uv run` (reinstala paddle e quebra o torch — usar `./.venv/bin/python` direto).
