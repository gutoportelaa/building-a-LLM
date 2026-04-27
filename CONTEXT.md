# CONTEXTO: Arquitetura de Dados DOM-PI — Diário Oficial dos Municípios do Piauí

Este documento descreve a arquitetura de dados e o roadmap técnico do pipeline de extração e
processamento do **Diário Oficial dos Municípios do Piauí (DOM-PI)**, cujo objetivo é construir
um corpus de documentos governamentais limpos, rastreáveis e sem redundâncias, pronto para
alimentar **Modelos Fundacionais (LLMs)** e sistemas **RAG (Retrieval-Augmented Generation)**.

---

## 1. Natureza dos Dados e Problema Central

### 1.1 O Diário Oficial como Edição Compartilhada

O DOM-PI publica edições consolidadas: **um mesmo arquivo PDF pode conter publicações de
múltiplos municípios**, diferentes entidades (Prefeitura, Câmara) e diversas categorias de atos
(Portarias, Editais, Decretos, Atas). Isso significa que:

- A mesma URL de PDF pode aparecer em dezenas de linhas do portal de busca para cidades diferentes.
- **Baixar o mesmo PDF repetidas vezes seria um desperdício** de banda e armazenamento.
- **Treinar o modelo com o mesmo conteúdo repetido** ("poisoning") comprometeria a qualidade do corpus.

A solução é uma **deduplicação em camadas**, separando o conceito de *aparição no diário*
do conceito de *arquivo físico* e do conceito de *conteúdo textual*.

### 1.2 PDFs Escaneados vs. Nativos

Um desafio técnico importante: **boa parte dos PDFs do DOM-PI são documentos escaneados** (imagens),
não PDFs com texto nativo. Isso significa que o texto extraído pelo PyMuPDF pode conter:

- **Blocos de texto limpo**: cabeçalhos, identificadores e conteúdo normativo real
- **Lixo de OCR**: caracteres sem sentido gerados pela camada OCR embutida no PDF

O pipeline usa um **filtro de qualidade OCR** (score 0.0–1.0) que avalia cada bloco de texto com
uma heurística composta: proporção alfanumérica, tamanho médio de palavras, presença de vocabulário
PT-BR e penalidade por sequências de caracteres especiais. Blocos abaixo do limiar (`MIN_OCR_QUALITY_SCORE = 0.35`)
são descartados antes da geração de Markdown e JSONL.

---

## 2. Arquitetura do Pipeline — 3 Módulos + 1 Suporte

O pipeline foi consolidado em **3 scripts operacionais independentes** (sem SQLite — apenas JSON/manifesto)
e **1 biblioteca de utilitários compartilhados**:

```
src/dompi_scraper/
├── pipeline.py          ← Etapa 1: Scraping paralelo (ThreadPoolExecutor)
├── download_pdfs.py     ← Etapa 2: Download controlado com manifesto SHA-256
├── processar_pdfs.py    ← Etapa 3: Extração PyMuPDF → Markdown + JSONL
├── shared_utils.py      ← Utilitários compartilhados (slugify, hash, classify_act)
└── extrator_marker.py   ← (em avaliação — extração via GPU/Marker, ver §10)
```

Cada script é **atômico e idempotente**: pode ser re-executado sem risco de corrupção.
A persistência é exclusivamente em **JSON + CSV + manifesto** — portável e sem dependência de servidor.

---

## 3. Fluxo End-to-End

```
Etapa 1           Etapa 2              Etapa 3
pipeline.py  →  download_pdfs.py  →  processar_pdfs.py
     ↓                ↓                     ↓
JSON dedup      manifest.json        .md + .jsonl
(por URL)       (SHA-256/PDF)        (Data Lake + Corpus)
```

---

## 4. Etapa 1 — Scraping de Metadados (`pipeline.py`)

**Responsabilidade:** Coletar metadados das publicações do portal DOM-PI sem baixar PDFs.

**Tecnologia:**
- `ThreadPoolExecutor` — cada cruzamento `[município × entidade]` roda em thread própria
- Cada thread cria **sessão HTTP e contexto de formulário independentes** (necessário pois
  os campos `hidden` do portal Scriptcase são vinculados à sessão)
- Paginação correta via **form F4** extraído do HTML da página anterior (não payload fixo)
- Retry com backoff exponencial (`2^attempt` segundos)

**Saídas:**
- `<saida>.json` — todos os registros brutos
- `<saida>_deduplicados.json` — PDFs únicos com `municipios_referenciados` (entrada da Etapa 2)
- `<saida>.csv` e `<saida>_deduplicados.csv` (opcional, `--so-json` omite)

**Teste isolado:**
```bash
# Teste rápido: 1 município, 5 publicações por entidade, 2 workers
uv run python src/dompi_scraper/pipeline.py \
    --municipio "Campo Maior" \
    --ano 2025 \
    --limite 5 \
    --max-workers 2 \
    --saida teste_scraping \
    --verbose

# Produção: Território Carnaubais completo
uv run python src/dompi_scraper/pipeline.py \
    --territorio-carnaubais \
    --ano 2025 \
    --max-workers 15 \
    --saida scraping_carnaubais_2025
```

---

## 5. Etapa 2 — Download de PDFs (`download_pdfs.py`)

**Responsabilidade:** Consumir o JSON deduplicado e baixar PDFs para disco local.

**Características:**
- **Incremental**: pula arquivos já existentes com status OK no manifesto
- **Integridade**: calcula SHA-256 pós-download e registra no manifesto
- **Checkpoint**: salva manifesto a cada 50 downloads (tolerância a interrupções)
- **Gravação atômica**: rename de arquivo temporário (sem manifesto corrompido)
- Retry com backoff exponencial (`2^attempt` segundos, até 3 tentativas)
- `--dry-run` para simulação sem downloads reais

**Saída:** `<output-dir>/download_manifest.json` — mapeamento `{md5_url → {path, sha256, status, metadados}}`

**Teste isolado:**
```bash
# Teste rápido: apenas 3 downloads
uv run python src/dompi_scraper/download_pdfs.py \
    --input teste_scraping_deduplicados.json \
    --output-dir teste_downloads/pdfs_arquivos \
    --limite 3 \
    --verbose

# Simulação sem download (valida registros)
uv run python src/dompi_scraper/download_pdfs.py \
    --input scraping_carnaubais_2025_deduplicados.json \
    --output-dir db_treino_carnaubais/pdfs_arquivos \
    --dry-run

# Produção: download completo
uv run python src/dompi_scraper/download_pdfs.py \
    --input scraping_carnaubais_2025_deduplicados.json \
    --output-dir db_treino_carnaubais/pdfs_arquivos
```

---

## 6. Etapa 3 — Processamento PDF → Markdown + JSONL (`processar_pdfs.py`)

**Responsabilidade:** Extrair texto estruturado dos PDFs baixados e gerar o corpus final.

**Pipeline interno:**
1. Lê manifesto de download (apenas entradas com `status: OK`)
2. Extrai blocos ricos via PyMuPDF (`font`, `size`, `bold`, `bbox`)
3. Aplica filtro de qualidade OCR (`compute_ocr_quality_score`, limiar 0.35)
4. **Modo padrão:** processa PDF inteiro como único documento
5. **Modo chunking (`--modo-chunking`):** detecta fronteiras de município via cabeçalhos
   (`PREFEITURA/CÂMARA DE <Cidade>`) e gera um `.md` independente por município detectado
6. Classifica tipo de ato por regex (`Portaria`, `Decreto`, `Lei`, `Edital`, etc.)
7. Gera Markdown hierárquico com frontmatter YAML (para RAG)
8. Armazena em Data Lake particionado `ano=YYYY/mes=MM/municipio=slug/`
9. Deduplicação textual por MD5 do conteúdo normalizado
10. Consolida JSONL para fine-tuning

**Saídas:**
- `<output-dir>/ano=YYYY/mes=MM/municipio=slug/<hash>.md` — Data Lake
- `<jsonl-output>` — texto puro para fine-tuning
- `<jsonl-output>.meta.jsonl` — texto + metadados para RAG
- `<output-dir>/registro_dedup.json` — controle de duplicatas textuais

**Teste isolado:**
```bash
# Calibração visual: analisa blocos de 3 PDFs sem processar
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl \
    --verbose-blocos-only 3

# Teste rápido: processa 5 PDFs com chunking por município
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest teste_downloads/pdfs_arquivos/download_manifest.json \
    --output-dir teste_datalake \
    --jsonl-output teste_corpus.jsonl \
    --limite 5 \
    --modo-chunking \
    --verbose

# Produção: processamento completo com chunking
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl \
    --modo-chunking
```

---

## 7. Execução Completa End-to-End

```bash
# Passo 1: Scraping
uv run python src/dompi_scraper/pipeline.py \
    --territorio-carnaubais --ano 2025 \
    --max-workers 15 \
    --saida scraping_carnaubais_2025

# Passo 2: Download
uv run python src/dompi_scraper/download_pdfs.py \
    --input scraping_carnaubais_2025_deduplicados.json \
    --output-dir db_treino_carnaubais/pdfs_arquivos

# Passo 3: Extração
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl \
    --modo-chunking
```

---

## 8. Formato Markdown para RAG

### 8.1 Frontmatter YAML

Todo arquivo `.md` gerado segue estritamente este formato:

```yaml
---
id_publicacao: "a8f9c2..."     # MD5 do conteúdo normativo (dedup key)
municipio: "Assunção do Piauí"
entidade: "Prefeitura Municipal"
tipo_ato: "Portaria"            # Classificado por regex no corpo
data_publicacao: "2025-03-15"   # ISO 8601
edicao: "5231"
sha256_pdf: "e3b0c44..."       # Integridade do arquivo fonte
url_origem: "https://..."
---
```

**Por que usar Frontmatter YAML?**

Bibliotecas de ingestão como **LangChain** (`UnstructuredMarkdownLoader`) e **LlamaIndex**
(`MarkdownReader`) leem o bloco `---` automaticamente e transformam as chaves em **metadados
do vetor**. Isso habilita **buscas híbridas** no RAG:

```python
# Exemplo LangChain: busca semântica + filtro por metadados
retriever.invoke(
    "licitações de asfalto",
    filter={"data_publicacao": {"$gte": "2025-01-01"}, "municipio": "Campo Maior"}
)
```

### 8.2 Hierarquia Textual (#, ##)

O corpo do Markdown é hierarquizado com base nos **atributos visuais** extraídos pelo PyMuPDF:

| Atributo Visual | Mapeamento Markdown |
|-----------------|---------------------|
| Font ≥ 14pt | `# Heading 1` (nome da entidade, cabeçalho) |
| Font ≥ 11pt + Bold | `## Heading 2` (título do ato: Portaria, Decreto) |
| Bold + "RESOLVE:" | `**RESOLVE:**` |
| "Art. 1º" | `**Art. 1º** - texto...` |
| Assinatura | `---` + `*Prefeito Municipal*` |
| Corpo normal | Parágrafo simples |

---

## 9. Data Lake Particionado

```
dados_brutos/
├── ano=2025/
│   ├── mes=01/
│   │   ├── municipio=assuncao_do_pi/
│   │   │   ├── a8f9c2e4.md
│   │   │   └── b3d1f7a9.md
│   │   ├── municipio=campo_maior/
│   │   │   └── c4e2a1b8.md
│   ├── mes=02/
│   │   └── ...
├── registro_dedup.json
```

---

## 10. JSONL para Fine-Tuning de LLMs

O arquivo `corpus_treino_dompi.jsonl` contém uma linha JSON por documento:

```json
{"text": "ESTADO DO PIAUÍ PREFEITURA MUNICIPAL DE ASSUNÇÃO DO PIAUÍ PORTARIA Nº 005/2025..."}
```

O arquivo `corpus_treino_dompi.meta.jsonl` inclui metadados para indexação auxiliar:

```json
{"text": "...", "metadata": {"id_publicacao": "abc...", "municipio": "...", "tipo_ato": "Portaria"}}
```

---

## 11. Deduplicação Textual Seletiva

O MD5 de deduplicação é calculado **exclusivamente sobre o corpo textual normalizado**:
- Remove espaços extras e quebras de linha múltiplas
- Converte para minúsculas
- Remove caracteres de controle Unicode
- **NÃO inclui**: URL, data de raspagem, metadados de frontmatter

---

## 12. Classificação de Tipo de Ato

| Padrão Detectado | Tipo Classificado |
|---|---|
| `PORTARIA` | Portaria |
| `DECRETO` | Decreto |
| `LEI Nº` | Lei |
| `EDITAL` | Edital |
| `LICITAÇÃO / PREGÃO / DISPENSA` | Licitação |
| `ATA DE SESSÃO / REGISTRO` | Ata |
| `CONTRATO / EXTRATO DE CONTRATO` | Contrato |
| `RESOLUÇÃO` | Resolução |
| `LRF / RELATÓRIO DE GESTÃO` | LRF |

---

## 13. Filtro de Qualidade OCR

`compute_ocr_quality_score()` avalia cada bloco em 4 dimensões:

1. **Proporção alfanumérica** (peso 45%): % de letras + dígitos + espaços vs total
2. **Tamanho médio de palavras** (peso 20%): palavras reais têm 3-15 chars, OCR tem 1-2
3. **Vocabulário PT-BR** (peso 25%): presença de palavras como "municipal", "prefeito", "resolve"
4. **Penalidade especial** (peso 10%): sequências de `~€#&@![]{}|<>` indicam lixo

Threshold: `MIN_OCR_QUALITY_SCORE = 0.35` — ajustável para corpus mais restrito (ex: 0.50).

**Calibração de thresholds:**
```bash
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl \
    --verbose-blocos-only 3
```

---

## 14. Chunking por Município (`--modo-chunking`)

PDFs do DOM-PI frequentemente são **edições consolidadas** contendo atos de múltiplos municípios.
Com `--modo-chunking`, o `processar_pdfs.py` detecta fronteiras de município via padrão:

```
PREFEITURA / CÂMARA DE <Cidade> [- PI]
```

A detecção filtra apenas blocos **visualmente destacados** (font ≥ 11pt ou negrito),
reduzindo falsos positivos no corpo do texto. Cada município detectado gera um `.md` independente
no Data Lake com o nome da cidade no caminho de particionamento.

> **Nota:** A detecção é heurística. PDFs com formatação irregular ou cabeçalhos muito próximos
> do corpo do texto podem gerar chunks "DESCONHECIDO". Revisar via `--verbose-blocos-only`.

---

## 15. Tecnologias Core

### PyMuPDF (fitz) — Motor de Extração

Extrai **atributos visuais** por bloco de texto:
- Família de fonte (Times-Bold, Helvetica, Arial-BoldMT)
- Tamanho em pontos (pt) e flags de estilo (negrito, itálico)
- Bounding box espacial (posição na página)

Esses atributos são essenciais para hierarquização do Markdown e detecção de cabeçalhos.

### Mimesis de Browser (Requests/BS4)

O portal DOM-PI usa formulário Scriptcase com campos hidden e sessões stateful.
Cada thread do scraper cria sessão própria, transita esses estados via `requests.Session`,
e usa `BeautifulSoup` para extrair `span_ids` sequenciais, links JavaScript e form F4
(controle de paginação stateful no servidor).

### Deduplicação por Hash (MD5)

Toda persistência usa MD5 como chave:
- `md5(url)` → ID do arquivo PDF (evita re-download)
- `md5(texto_normalizado)` → ID do conteúdo textual (evita re-ingestão)

### extrator_marker.py (em avaliação)

Módulo separado em fase de testes e otimização para extração via GPU (Marker).
**Não integrado ao pipeline principal ainda.** Ver documentação interna do módulo.
