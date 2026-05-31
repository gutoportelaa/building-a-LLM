# Benchmark Suite: DOM-PI Scraper Performance Comparison

Relatório comparativo de desempenho para coleta e download de documentos usando diferentes abordagens:

- **requests + BeautifulSoup** (síncrono com ThreadPool opcional)
- **aiohttp + asyncio** (assíncrono)
- **Selenium** (híbrido: navegação via Selenium + download via requests)
- **Selenium** (completo: todo o workflow pelo navegador)

## Estrutura dos Arquivos

```
src/dompi_scraper/bench/
├── __init__.py                 # Pacote
├── common_downloader.py        # Instrumentação, logging e utilitários compartilhados
├── requests_bench.py           # Implementação com requests (sync + threaded)
├── aiohttp_bench.py            # Implementação com aiohttp (async)
├── selenium_bench.py           # Implementação com Selenium (hybrid + full)
├── runner_bench.py             # Orquestrador genérico (não usado diretamente)
├── run_benchmark.py            # Script principal para executar benchmarks
├── analyze_results.py          # Análise de resultados e geração de gráficos
└── README.md                   # Este arquivo
```

## Instalação de Dependências

As dependências principais já estão no `pyproject.toml` do projeto:
- `requests`
- `beautifulsoup4`
- `aiohttp`
- `selenium`

Dependências adicionais opcionais:
```bash
pip install psutil pandas matplotlib
```

Para Selenium, é necessário ter o Chrome/Chromium instalado e o ChromeDriver compatível.

## Uso Básico

### 1. Execução de Benchmarks

**Quick test com 50 URLs:**
```bash
cd src/dompi_scraper/bench
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --limit 50 \
  --approaches requests_sync aiohttp_async \
  --concurrency 1 5
```

**Teste completo com dataset padrão:**
```bash
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --approaches requests_sync aiohttp_async selenium_hybrid \
  --concurrency 1 5 10 20 \
  --runs 3 \
  --warmups 1 \
  --output bench_results
```

**Comparar apenas duas abordagens:**
```bash
python run_benchmark.py \
  --dataset <arquivo.json> \
  --approaches requests_sync aiohttp_async \
  --concurrency 5 10 15 \
  --runs 5
```

### 2. Análise de Resultados

Após a execução dos benchmarks:

```bash
# Gerar relatório completo com gráficos
python analyze_results.py bench_results/results_*.csv \
  --output benchmark_report.md \
  --plots-dir plots

# Sem gráficos (se matplotlib não estiver disponível)
python analyze_results.py bench_results/results_*.csv \
  --output benchmark_report.md \
  --no-plots
```

## Argumentos de Linha de Comando

### `run_benchmark.py`

```
--dataset FILE                   [OBRIGATÓRIO] Arquivo JSON com URLs
--approaches {requests_sync,aiohttp_async,selenium_hybrid,selenium_full}
                                 Abordagens a testar (padrão: requests_sync, aiohttp_async, selenium_hybrid)
--concurrency N [N ...]          Níveis de concorrência a testar (padrão: 1 5 10)
--timeout SECONDS                Timeout para requisições HTTP (padrão: 30)
--runs N                         Número de rodadas de medição (padrão: 2)
--warmups N                      Número de warmups (descartados) (padrão: 1)
--limit N                        Limitar a N URLs (útil para testes rápidos)
--output DIR                     Diretório de saída (padrão: bench_results)
```

### `analyze_results.py`

```
results CSV_FILE [CSV_FILE ...]  Arquivo(s) CSV com resultados
--output FILE                    Arquivo Markdown de saída (padrão: benchmark_report.md)
--plots-dir DIR                  Diretório para salvar gráficos (padrão: plots)
--no-plots                       Pular geração de gráficos
```

## Formato de Dados

### Entrada (JSON)

Esperado um arquivo JSON com lista de URLs ou objetos contendo `pdf_url`/`url`:

```json
[
  {"pdf_url": "https://example.com/doc1.pdf", ...},
  {"pdf_url": "https://example.com/doc2.pdf", ...},
  ...
]
```

### Saída (CSV)

O CSV de resultados contém colunas:

| Campo | Descrição |
|-------|-----------|
| `run_id` | Identificador único da rodada |
| `approach` | Abordagem usada (requests_sync, aiohttp_async, etc.) |
| `concurrency` | Nível de concorrência |
| `url` | URL do documento |
| `status` | Status do download (success, timeout, http_error, etc.) |
| `success` | Booleano de sucesso |
| `wall_time_s` | Tempo total em segundos |
| `bytes_downloaded` | Bytes transferidos |
| `sha256` | Hash SHA256 do arquivo |
| `cpu_user_s` | Tempo CPU em user mode |
| `cpu_sys_s` | Tempo CPU em sys mode |
| `memory_rss_mb` | Memória RSS em MB |
| `timestamp_start` | Timestamp ISO 8601 início |
| `timestamp_end` | Timestamp ISO 8601 fim |
| `http_status_code` | Status HTTP (se aplicável) |
| `error_msg` | Mensagem de erro (se houver) |

## Exemplo Completo de Workflow

```bash
# 1. Ir para o diretório
cd ~/Documents/building-a-LLM/src/dompi_scraper/bench

# 2. Executar benchmarks (50 URLs, 3 abordagens, 2 concorrências, 2 rodadas)
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --approaches requests_sync aiohttp_async selenium_hybrid \
  --concurrency 1 10 \
  --limit 50 \
  --runs 2 \
  --output bench_results

# 3. Analisar resultados
python analyze_results.py bench_results/results_*.csv \
  --output benchmark_report.md \
  --plots-dir plots

# 4. Visualizar relatório
cat benchmark_report.md
ls plots/
```

## Métricas Coletadas

Para cada download, o benchmark coleta:

1. **Tempo** (wall time)
   - Tempo total desde início até fim
   - Inclui overhead de conexão, DNS, TLS, transferência, escrita em disco

2. **Recursos**
   - CPU time (user + sys)
   - Memória RSS (resident set size)

3. **Transferência**
   - Bytes baixados
   - Hash SHA256 para verificação

4. **HTTP**
   - Status code
   - Sucesso/falha

## Interpretação dos Resultados

### Throughput (docs/sec)
- Métrica de produção: quantos documentos/segundo
- **Maior é melhor**
- Afetado por concorrência e latência de rede

### Tempo médio/mediano
- Latência por documento
- Importante para modo single-threaded
- **Menor é melhor**

### Taxa de sucesso
- Percentual de downloads bem-sucedidos
- Deve estar > 95% para todas as abordagens
- Pode cair em alta concorrência por rate-limiting

### Consumo de memória
- RSS médio e máximo
- Importante para ambientes com restrição de recursos
- aiohttp geralmente tem menor footprint

### Uso de CPU
- Tempo de CPU user + sys
- Pode indicar gargalos (parsing, compressão)
- Selenium tipicamente tem overhead mais alto

## Observações Importantes

### Selenium
- Muito mais lento que requests/aiohttp (rendering JS)
- Modo **hybrid** separa o custo de rendering do download
- Modo **full** mede o overhead total do navegador
- Concorrência limitada (resource-intensive)

### aiohttp
- Escalabilidade superior em alta concorrência
- Melhor eficiência de CPU que threading
- Pode ter problemas com proxies/firewalls

### requests + ThreadPool
- Simples e robusto
- Escalabilidade limitada por threads (GIL do Python)
- Bom para uso geral

### Recomendações Práticas

1. **Para coleta rápida de muitos documentos**: `aiohttp_async` com `concurrency=20-50`
2. **Para produção estável**: `requests_sync` com `concurrency=5-10`
3. **Para conteúdo dinâmico/JS**: `selenium_hybrid` com `concurrency=2-5`
4. **Para benchmarking interno**: incluir `selenium_hybrid` para isolar custos

## Troubleshooting

**"Chrome/Chromium not found"**
- Instalar Chrome/Chromium
- Ou configurar `webdriver.Chrome(options=options)` com caminho explícito

**"Connection reset by peer"**
- Reduzir `--concurrency`
- Aumentar `--timeout`

**"All runs failed"**
- Verificar se as URLs estão acessíveis
- Checar conectividade de rede
- Aumentar timeout

**Memória insuficiente (Selenium)**
- Reduzir `--limit` para menos URLs por rodada
- Usar `--concurrency 1` ou `2`

## Customização

Para adicionar uma nova abordagem:

1. Criar novo arquivo (e.g., `custom_bench.py`)
2. Implementar função `benchmark_func(urls, concurrency, timeout) -> List[BenchmarkResult]`
3. Adicionar ao dicionário `factories` em `run_benchmark.py`
4. Invocar via `--approaches custom_approach`

## Leitura Adicional

- [Common Downloader API](common_downloader.py) - Detalhes de classe BenchmarkResult
- [CSV Analysis](analyze_results.py) - Métodos de análise estatística
- Logs detalhados em `bench_results/*.log`
