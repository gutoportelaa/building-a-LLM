# Projeto DOM-PI Scraper

Repositório dedicado à automação da coleta e processamento de publicações do Diário Oficial dos Municípios do Piauí (DOM-PI), com escopo primário focado no Território de Desenvolvimento dos Carnaubais. O sistema realiza extração de metadados, download de publicações, processamento de PDFs com estratégias de deduplicação incremental e conversão estruturada.

## Visão Arquitetural

A arquitetura do projeto foi consolidada em uma pipeline unificada (*end-to-end*) sob a interface `pipeline.py`. Isso minimiza redundâncias operacionais, reduz alocações de I/O de rede desnecessárias e viabiliza um fluxo coerente:

1. **Scraping Inicial:** Estabelecimento de sessões HTTP persistentes (`requests.Session`) para contornar protocolos restritivos do backend *Scriptcase*.
2. **Parsing Estrutural:** Uso do `BeautifulSoup` na decodificação de DOM Trees fragmentadas e extração orientada por Expressões Regulares (Regex) em seletores ocultos.
3. **Deduplicação de Artefatos:** Verificação nativa em memória no momento da varredura, prevenindo dowloads paralelos para arquivos de *hash/url* homólogo.
4. **Conversão Textual (Markdown):** Transformação assíncrona baseada em heurísticas usando as bibliotecas associadas (`markitdown`, `pypdf`).
5. **Relatórios Consolidativos:** Geração de *logs* vitais e emissão de um relatório executivo apontando volume das amostras, sucessos e ocorrências excepcionais (timeouts).

## Estrutura do Repositório

- `src/dompi_scraper/pipeline.py`: O núcleo de execução contendo orquestrador principal, crawling, parsing web e manipulação local (A Pipeline Unificada).
- `src/dompi_scraper/schema_utils.py`: Transacionador de esquemas para manipulação persistente e padronizada das tabelas `CSV`.
- `src/dompi_scraper/shared_utils.py`: Auxiliares computacionais abrangentes (tratadores de *slugs* e formatadores de sintaxe).
- `docs/research/`: Camada contextual da engenharia reversa executada sobre o ecossistema DOM-PI.
- `CONTEXT.md`: Regras de negócio essenciais e fundamentos didáticos da biblioteca adotada para instrução técnica de novos mantenedores.

## Passo-a-Passo: Inicializando o Ambiente (Setup)

O projeto é gerenciado rigorosamente pelo sistema **uv** (para empacotamento ultrarrápido). As etapas de configuração para replicação do ambiente de desenvolvimento são as seguintes:

### Pré-Requisitos
1. **Python 3.12** ou superior estar contido no seu `$PATH`.
2. O gerenciador **uv** instalado na sua máquina (`curl -LsSf https://astral.sh/uv/install.sh | sh` no unix).

### Etapa 1: Sincronização do Ambiente (`.venv`)
Execute os comandos na raiz da infraestrutura para validar o lockfile versionado e construir o virtual environment isolado:

```bash
uv sync
```
*O uso do comando `uv sync` certificará as dependências obrigatórias: `requests`, `beautifulsoup4`, `markitdown` e `pypdf` e criará o diretório `.venv` silenciosamente.*

### Etapa 2: Validar Instalação
Garanta que as bibliotecas e interpretadores estejam saudáveis consultando a invocação nativa do `help`:

```bash
uv run python src/dompi_scraper/pipeline.py --help
```

## Execução da Pipeline Principal

A execução primária se dá por ativação do script unificado e se modela através dos *flags* informados. Toda execução criará nativamente diretórios de metadados (`/pdfs_arquivos`, `/markdowns`), *logs* estruturados no padrão ANSI (`scraper_operacional.log`) e resultados de performance.

### 1. Teste Focado por Entidade/Município
Uma extração em amostra baseada em um município delimitado para fins de validação paralela e testes rápidos:

```bash
uv run python src/dompi_scraper/pipeline.py --municipio "Campo Maior" --inicio "01/01/2025" --fim "31/12/2025" --output "./resultado_local"
```

### 2. Job Executivo (Território Carnaubais em Lote)
Para acionar a instrução abrangendo os 16 Municípios rastreados pelo polo de negócios de forma nativa e paralela:

```bash
uv run python src/dompi_scraper/pipeline.py --territorio-carnaubais --inicio "01/01/2025" --fim "31/12/2025" --output "./saida_carnaubais"
```

## Diretrizes e Políticas de Persistência

- **Carga Massiva (Cold Storage):** Nenhum arquivo de carga de processamento `.pdf`, transcrições espessas de `.md` ou logs contendo mais de mil interações não sumarizadas deverão transitar pelo repositório base. Use explicitamente os apontamentos em `.gitignore`.
- **Evoluções da Malla:** Se mudanças nas regras do fornecedor (Scripcase/DOM-PI) inviabilizarem o funcionamento orgânico dos extratores (blocos BeautifulSoup), atente-se às documentações geradas nos metadados salvos ou revise os fundamentos explicitados em `CONTEXT.md`.