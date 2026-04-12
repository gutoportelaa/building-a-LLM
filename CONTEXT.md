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

## 2. Arquitetura Relacional (SQLite — Star Schema)

O núcleo do pipeline hoje reside num banco `SQLite` local (`dompi_knowledge_base.sqlite`),
estruturado em 3 tabelas relacionais que modelam cada camada do problema:

```
fato_publicacoes  ──FK──>  dim_documentos_pdf  ──FK──>  dim_extracoes_texto
(aparições)                (arquivos únicos)             (textos únicos)
```

### 2.1 `fato_publicacoes` — Tabela Fato (Aparições)

Registra **cada ocorrência de um ato no Diário Oficial**. É o espelho fiel do portal de busca:
se uma licitação for publicada 3 vezes consecutivas, existirão 3 linhas aqui — com
**Entidade Fonte** (Prefeitura, Câmara) e metadados extraídos da URL por regex
(número da edição, código interno do município, página).

> Esta tabela não sabe se o PDF foi baixado. Ela apenas registra que *aquela publicação existiu*.

### 2.2 `dim_documentos_pdf` — Deduplicação de Arquivo Físico

Controla os arquivos em disco. A chave primária é o **MD5 da URL do PDF**: se múltiplas
publicações em `fato_publicacoes` (de cidades diferentes!) apontam para o mesmo arquivo,
o download ocorre **uma única vez**. As próximas inserções reutilizam o registro existente.

Esta camada resolve o gargalo de rede e armazenamento: edições do DOM-PI podem ser grandes
(centenas de MB), mas como são compartilhadas, o volume total em disco escala de forma
controlada independentemente do número de municípios indexados.

> O campo `path_local` aponta para o PDF em disco. O `status_download` pode ser `OK` ou `FAILED`.

### 2.3 `dim_extracoes_texto` — Deduplicação Textual (Corpus RAG)

A *coroa de joia* do pipeline. O texto extraído do PDF (via OCR/MarkItDown) recebe um
**MD5 do conteúdo normalizado** como chave primária. Dois PDFs com conteúdo idêntico —
mesmo que com nomes diferentes — resultam num único registro aqui.

Esta tabela é o corpus final para ingestão em LLMs e vector databases (Qdrant, Chroma, Pinecone).

---

## 3. Pipeline Modularizado — Scripts Isolados

O pipeline foi decomposto em **scripts atômicos e independentes** para facilitar
validação humana, debug e re-execução parcial de cada estágio.

### Estágio 1 — Scraping de Metadados ✅

**Script:** `scraper_isolado.py`

Coleta metadados das publicações do DOM-PI sem baixar PDFs e sem acessar SQLite.
Gera JSON e CSV com os registros coletados, realiza lógica de upsert incremental
e deduplicação pré-download (agrupa URLs únicas).

```bash
# Território Carnaubais completo (16 municípios × 2 entidades, sem limite)
uv run python src/dompi_scraper/scraper_isolado.py \
    --territorio-carnaubais --ano 2025 --limite 1000000 --verbose
```

**Saídas:**
- `scraping_carnaubais_2025.json` — registros brutos (~12.372 publicações)
- `scraping_carnaubais_2025_deduplicados.json` — URLs únicas com metadados enriquecidos
- Relatório de discrepâncias e distribuição por município/entidade

### Estágio 2 — Download de PDFs ✅

**Script:** `download_pdfs.py`

Consome o JSON deduplicado e baixa PDFs para disco local com:
- Controle incremental (pula arquivos já existentes)
- Hash SHA-256 pós-download para integridade
- Manifesto JSON com mapeamento `{url → path, sha256, status, metadados}`
- Retry com backoff exponencial
- Flags: `--dry-run`, `--limite N`

```bash
# Download dos primeiros 10 PDFs para teste
uv run python src/dompi_scraper/download_pdfs.py \
    --input scraping_carnaubais_2025_deduplicados.json \
    --output-dir db_treino_carnaubais/pdfs_arquivos \
    --limite 10
```

**Saída:** `download_manifest.json` — manifesto de integridade

### Estágio 3 — Processamento: PDF → Markdown + JSONL ✅

**Script:** `processar_pdfs.py`

Lê PDFs baixados, extrai texto estruturado via PyMuPDF e gera:

1. **Markdown com Frontmatter YAML** — para ingestão em RAG (LangChain/LlamaIndex)
2. **JSONL** — para fine-tuning de LLMs
3. **Registro de deduplicação** — hash MD5 do conteúdo normativo

```bash
# Processamento com análise detalhada de blocos
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl \
    --verbose-blocos-only 3

# Processamento completo
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl
```

**Saídas:**
- `dados_brutos/ano=YYYY/mes=MM/municipio=slug/hash.md` — Data Lake particionado
- `corpus_treino_dompi.jsonl` — texto puro para fine-tuning
- `corpus_treino_dompi.meta.jsonl` — texto + metadados para RAG
- `dados_brutos/registro_dedup.json` — controle de duplicatas textuais

---

## 4. Formato Markdown para RAG

### 4.1 Frontmatter YAML

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

### 4.2 Hierarquia Textual (#, ##)

O corpo do Markdown é hierarquizado com base nos **atributos visuais** extraídos pelo PyMuPDF:

| Atributo Visual | Mapeamento Markdown |
|-----------------|---------------------|
| Font ≥ 14pt | `# Heading 1` (nome da entidade, cabeçalho) |
| Font ≥ 11pt + Bold | `## Heading 2` (título do ato: Portaria, Decreto) |
| Bold + "RESOLVE:" | `**RESOLVE:**` |
| "Art. 1º" | `**Art. 1º** - texto...` |
| Assinatura | `---` + `*Prefeito Municipal*` |
| Corpo normal | Parágrafo simples |

Essa hierarquia permite que o `MarkdownHeaderTextSplitter` (LangChain) fatia o texto
respeitando os limites semânticos — o título de uma portaria nunca fica separado do
seu artigo 1º em vetores diferentes.

---

## 5. Data Lake Particionado

Os arquivos `.md` são armazenados fisicamente em disco com particionamento estilo Data Lake:

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

**Vantagem:** Se no futuro trocar de Vector DB (Qdrant → Pinecone) ou alterar a
estratégia de chunking, os dados brutos permanecem intactos e de fácil acesso.
A re-ingestão é trivial — basta iterar nos diretórios.

---

## 6. JSONL para Fine-Tuning de LLMs

O arquivo `corpus_treino_dompi.jsonl` contém uma linha JSON por documento, com
apenas o texto limpo (sem metadados, sem frontmatter):

```json
{"text": "ESTADO DO PIAUÍ PREFEITURA MUNICIPAL DE ASSUNÇÃO DO PIAUÍ PORTARIA Nº 005/2025 Dispõe sobre..."}
{"text": "CÂMARA MUNICIPAL DE CAMPO MAIOR DECRETO Nº 012/2025 DECRETA: Art. 1º ..."}
```

O arquivo `corpus_treino_dompi.meta.jsonl` adicionalmente inclui metadados no
campo `metadata` para indexação auxiliar:

```json
{"text": "...", "metadata": {"id_publicacao": "abc...", "municipio": "...", "tipo_ato": "Portaria"}}
```

---

## 7. Deduplicação Textual Seletiva

### Hash sobre Conteúdo Normativo Puro

O MD5 de deduplicação é calculado **exclusivamente sobre o corpo textual normalizado**:
- Remove espaços extras e quebras de linha múltiplas
- Converte para minúsculas
- Remove caracteres de controle Unicode
- **NÃO inclui**: URL, data de raspagem, metadados de frontmatter

Isso garante que dois PDFs com o mesmo conteúdo normativo (mesmo publicados em datas
ou edições diferentes) produzam o mesmo hash e sejam registrados apenas uma vez.

### Republicação por Incorreção (Upsert)

Documentos governamentais frequentemente sofrem "republicação por incorreção".
O registro de deduplicação (`registro_dedup.json`) mantém o mapeamento
`{hash → {path, municipio, data, url}}`. Em caso de colisão, o registro mais
recente sobrescreve o anterior — garantindo que o RAG entregue a versão corrigida.

---

## 8. Classificação de Tipo de Ato

O tipo de ato é classificado por **regex sobre o corpo do texto**, não confiando
apenas no campo `categoria` do scraping (que pode ser genérico). A lista de padrões
cobre os tipos mais frequentes no DOM-PI:

| Padrão Detectado | Tipo Classificado |
|-------------------|-------------------|
| `PORTARIA` | Portaria |
| `DECRETO` | Decreto |
| `LEI Nº` | Lei |
| `EDITAL` | Edital |
| `LICITAÇÃO / PREGÃO / DISPENSA` | Licitação |
| `ATA DE SESSÃO / REGISTRO` | Ata |
| `CONTRATO / EXTRATO DE CONTRATO` | Contrato |
| `RESOLUÇÃO` | Resolução |
| `LRF / RELATÓRIO DE GESTÃO` | LRF |

Se nenhum padrão for detectado, o sistema usa como fallback o campo `categoria`
vindo do JSON de scraping.

---

## 9. Filtro de Qualidade OCR

### Heurísticas de Score

Como muitos PDFs do DOM-PI são escaneados, o pipeline implementa um score de qualidade
(`compute_ocr_quality_score`) baseado em 4 dimensões:

1. **Proporção alfanumérica** (peso 45%): % de letras + dígitos + espaços vs total
2. **Tamanho médio de palavras** (peso 20%): palavras reais têm 3-15 chars, OCR tem 1-2
3. **Vocabulário PT-BR** (peso 25%): presença de palavras como "municipal", "prefeito", "resolve"
4. **Penalidade especial** (peso 10%): sequências de `~€#&@![]{}|<>` indicam lixo

O threshold `MIN_OCR_QUALITY_SCORE = 0.35` é conservador — pode ser ajustado para cima
(ex: 0.50) para um corpus mais limpo mas com potencial perda de conteúdo limítrofe.

### Modo Verbose-Blocos (Calibração)

Para ajustar os thresholds, o script oferece análise visual detalhada:

```bash
# Despeja análise de 3 PDFs sem processar (apenas calibração)
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl \
    --verbose-blocos-only 3
```

Saída mostra por bloco: negrito, tamanho de fonte, score OCR (Q=0.65 ✓OK), candidatos
a heading, e distribuição estatística de tamanhos.

---

## 10. Tecnologias Core

### PyMuPDF (fitz) — Motor de Extração

O PyMuPDF é o motor primário de extração textual. Diferente do MarkItDown (genérico),
o PyMuPDF fornece **atributos visuais** por bloco de texto:
- Família de fonte (Times-Bold, Helvetica, Arial-BoldMT)
- Tamanho em pontos (pt)
- Flags de estilo (negrito, itálico, sublinhado)
- Bounding box espacial (posição na página)

Esses atributos são essenciais para a hierarquização do Markdown e para a detecção
de cabeçalhos de municípios em PDFs compartilhados.

### Mimesis de Browser (Requests/BS4)

O portal DOM-PI usa um sistema de formulário com campos hidden e sessões stateful.
O pipeline transita esses estados enviando headers corretos (`User-Agent`, `Referer`)
via `requests.Session`, enquanto o `BeautifulSoup` extrai os `span_ids` sequenciais
e decodifica os links JavaScript para obter as URLs reais dos PDFs.

### Deduplicação por Hash (MD5)

Todo objeto persistido — arquivo PDF ou texto extraído — recebe um hash MD5 como chave.
Isso transforma o `INSERT OR IGNORE` do SQLite numa barreira eficiente contra duplicatas,
sem necessidade de queries de verificação prévia.

### SQLite como Hub Intermediário

Para a fase de captura e deduplicação cruzada entre entidades, o SQLite oferece
integridade relacional nativa sem overhead de infra. O banco pode ser trivialmente
exportado para Parquet ou ingerido em vector databases (Qdrant, Pinecone, ChromaDB)
na transição para produção.

---

## 11. Execução Completa (Passo a Passo)

### Passo 1: Scraping de Metadados
```bash
uv run python src/dompi_scraper/scraper_isolado.py \
    --territorio-carnaubais --ano 2025 --limite 1000000
```
Saída: `scraping_carnaubais_2025_deduplicados.json`

### Passo 2: Download dos PDFs
```bash
uv run python src/dompi_scraper/download_pdfs.py \
    --input scraping_carnaubais_2025_deduplicados.json \
    --output-dir db_treino_carnaubais/pdfs_arquivos
```
Saída: `db_treino_carnaubais/pdfs_arquivos/download_manifest.json`

### Passo 3: Processamento → Markdown + JSONL
```bash
uv run python src/dompi_scraper/processar_pdfs.py \
    --manifest db_treino_carnaubais/pdfs_arquivos/download_manifest.json \
    --output-dir dados_brutos \
    --jsonl-output corpus_treino_dompi.jsonl
```
Saídas:
- `dados_brutos/ano=*/mes=*/municipio=*/*.md` — Data Lake
- `corpus_treino_dompi.jsonl` — Corpus para LLM
- `corpus_treino_dompi.meta.jsonl` — Corpus com metadados para RAG

O banco `dompi_knowledge_base.sqlite` e os logs ficam em `./db_treino_carnaubais/`.
