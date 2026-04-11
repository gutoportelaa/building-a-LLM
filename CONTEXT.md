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

## 3. Roadmap Técnico — Fases do Pipeline

### Fase 1 — Scraping e Indexação de Metadados ✅ (em andamento)
- Crawler via `requests.Session` + `BeautifulSoup` mimetiza o formulário de busca do DOM-PI.
- Itera sobre `[Município × Entidade]` coletando metadados das publicações.
- Persiste metadados em `fato_publicacoes` e registra o PDF em `dim_documentos_pdf`.
- O download dos PDFs é **opcional e controlado pelo pipeline** (pode ser inibido para testar apenas o scraping).
- Flag `--limite X` permite validar a arquitetura sem estourar disco/rede.

### Fase 2 — Extração Textual com Deduplicação de Conteúdo (planejado)
- Para cada PDF em `dim_documentos_pdf` com `status_download = OK`, rodar `MarkItDown`.
- Normalizar e hashear o texto extraído → inserir em `dim_extracoes_texto` com `INSERT OR IGNORE`.
- A deduplicação por hash de texto garante que o mesmo conteúdo não entra duas vezes no corpus,
  mesmo que venha de arquivos ou cidades diferentes.

### Fase 3 — Atribuição de Páginas por Cidade (planejado)
- Como um PDF pode conter publicações de múltiplos municípios, a Fase 3 identificará
  as **páginas específicas de cada cidade** dentro do arquivo.
- Estratégia: cruzar os metadados de `fato_publicacoes` (campo `pagina_codigo_metadata`)
  com o conteúdo textual extraído para localizar os trechos correspondentes a cada município.
- Resultado: enriquecer `dim_extracoes_texto` com campos `pagina_inicio` e `pagina_fim` por cidade.

### Fase 4 — Fatiamento de PDFs por Cidade (planejado)
- Com as páginas identificadas, usar uma biblioteca de manipulação de PDF
  (ex: `pypdf`, `pdfplumber`) para **cortar o PDF original** em sub-arquivos.
- Cada sub-arquivo conterá **apenas as páginas com publicações daquele município**.
- Os arquivos fatiados são armazenados localmente e referenciados no banco,
  permitindo ao corpus final ter documentos oficiais **exclusivos por cidade** — sem páginas irrelevantes.
- Isso reduz drasticamente o ruído nos embeddings do sistema RAG.

---

## 4. Tecnologias Core

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

## 5. Execução do Pipeline

### Teste Controlado — Território Carnaubais (16 municípios, limite por entidade)
```bash
uv run python src/dompi_scraper/pipeline.py --territorio-carnaubais --limite 5
```

### Teste Isolado — Apenas Scraping (sem download de PDF, sem SQLite)
```bash
uv run python src/dompi_scraper/scraper_isolado.py \
    --municipio "Assuncao do Pi" --entidade Prefeitura --ano 2025 --limite 10
```

### Saída esperada no `stdout`
```
[INFO] -> Avaliando: [Cidade: Assuncao do Pi] | [Entidade: Prefeitura]
[INFO]    - Salvos 5 registros para Assuncao do Pi / Prefeitura.
```

O banco `dompi_knowledge_base.sqlite` e os logs ficam em `./db_treino_carnaubais/`.
