# GUIA DE EXECUÇÃO - Relatório Comparativo de Desempenho

## 🎯 Objetivo

Comparar o desempenho de três abordagens para coleta e download de documentos PDF:

1. **requests + BeautifulSoup** (síncrono com ThreadPool)
2. **aiohttp + asyncio** (assíncrono)
3. **Selenium** (navegação dinâmica + download)

Medindo: latência, throughput, uso de CPU/memória, e taxa de sucesso.

---

## 📋 Pré-requisitos

```bash
# Verificar ambiente
python validate_env.py

# Se alguma dependência faltar:
pip install requests beautifulsoup4 aiohttp selenium psutil pandas matplotlib
```

---

## ⚡ Início Rápido (5 minutos)

### Teste rápido com 50 URLs:

```bash
cd ~/Documents/building-a-LLM/src/dompi_scraper/bench

# Opção 1: Via script shell
chmod +x quick_test.sh
./quick_test.sh 50

# Opção 2: Via Python
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --limit 50 \
  --approaches requests_sync aiohttp_async \
  --concurrency 1 5 \
  --output test_results
```

Após execução:
- Ver relatório: `cat test_results/report.md`
- Ver gráficos: `ls test_results/plots/`

---

## 📊 Execução Completa (30-60 minutos)

### Teste padrão com todos os perfis:

```bash
# 1️⃣ Teste rápido (10 min)
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --approaches requests_sync aiohttp_async selenium_hybrid \
  --concurrency 1 5 10 \
  --limit 100 \
  --runs 2 \
  --output results_light
```

```bash
# 2️⃣ Teste padrão (30 min) - RECOMENDADO
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --approaches requests_sync aiohttp_async selenium_hybrid selenium_full \
  --concurrency 1 5 10 20 \
  --limit 200 \
  --runs 3 \
  --warmups 1 \
  --output results_standard
```

```bash
# 3️⃣ Análise e relatório final
python analyze_results.py results_standard/results_*.csv \
  --output results_standard/REPORT.md \
  --plots-dir results_standard/plots
```

---

## 🔧 Argumentos Principais

```
--dataset FILE                    [OBRIGATÓRIO] Arquivo JSON com URLs

--approaches APPROACH [...]       Quais testar (padrão: requests_sync aiohttp_async)
                                  Opções: requests_sync, aiohttp_async, 
                                         selenium_hybrid, selenium_full

--concurrency N [N ...]          Níveis de concorrência (padrão: 1 5 10)
                                  Exemplo: 1 5 10 20 50

--limit N                         Limitar a N URLs (útil para testes rápidos)
                                  Padrão: usar todas

--runs N                          Quantas rodadas de medição (padrão: 2)
                                  Recomendado: 3-5 para resultados estáveis

--warmups N                       Quantas rodadas descartadas (padrão: 1)
                                  Para estabilizar antes de medir

--timeout SECS                    Timeout HTTP em segundos (padrão: 30)

--output DIR                      Diretório de saída (padrão: bench_results)
```

---

## 📈 Exemplos Práticos

### A. Comparar apenas requests vs aiohttp:

```bash
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --approaches requests_sync aiohttp_async \
  --concurrency 1 5 10 20 \
  --limit 200 \
  --runs 5 \
  --output comparison_sync_vs_async
```

### B. Medir overhead de Selenium:

```bash
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --approaches selenium_hybrid selenium_full \
  --concurrency 1 2 \
  --limit 50 \
  --runs 3 \
  --output selenium_overhead
```

### C. Teste de carga com aiohttp:

```bash
python run_benchmark.py \
  --dataset ../../../dados/scraping_results/scraping_carnaubais_2025_deduplicados.json \
  --approaches aiohttp_async \
  --concurrency 1 5 10 20 50 100 \
  --limit 500 \
  --runs 5 \
  --output stress_test_aiohttp
```

---

## 📝 Interpretando Resultados

### Arquivo CSV (bench_results/results_*.csv)

Cada linha é um download com:
- `approach`: qual método (requests_sync, aiohttp_async, etc.)
- `concurrency`: quantas conexões simultâneas
- `wall_time_s`: tempo total em segundos
- `bytes_downloaded`: tamanho do arquivo
- `cpu_user_s`, `cpu_sys_s`: tempo de CPU usado
- `memory_rss_mb`: memória consumida
- `success`: true/false
- `error_msg`: detalhes de falhas

### Relatório Markdown (REPORT.md)

Seções principais:

1. **Summary Statistics** - por cenário (abordagem + concorrência)
   - Taxa de sucesso %
   - Tempo médio/mediano por documento
   - Throughput (docs/sec)
   - Consumo de CPU e memória

2. **Comparison by Approach** - agregado por abordagem
   - Qual é mais rápido, eficiente, etc.

3. **Gráficos** (plots/)
   - `wall_time_boxplot.png` - distribuição de latências
   - `throughput_bar.png` - produção em docs/sec
   - `memory_bar.png` - consumo de memória
   - `success_rate.png` - taxa de sucesso %

### Métricas-Chave

| Métrica | Melhor | Interpretação |
|---------|--------|---------------|
| **Throughput (docs/sec)** | Maior | Produção por segundo |
| **Tempo médio (s)** | Menor | Latência por documento |
| **Taxa de sucesso %** | Maior | Robustez e confiabilidade |
| **Memória (MB)** | Menor | Eficiência de recursos |
| **CPU time (s)** | Menor | Custo computacional |

---

## 💡 Recomendações de Uso

### Para coleta em produção:

**Alto volume, recursos limitados:**
```
aiohttp_async com concurrency=20-50
→ Melhor throughput, menor memória
```

**Estabilidade > velocidade:**
```
requests_sync com concurrency=5-10 + ThreadPool
→ Mais robusto, fácil de debugar
```

**Conteúdo com JavaScript:**
```
selenium_hybrid com concurrency=2-4
→ Renderiza JS, mas download eficiente
```

---

## 🔍 Troubleshooting

### "Connection refused / reset"
```bash
# Reduzir concorrência
python run_benchmark.py ... --concurrency 1 5
```

### "Timeout"
```bash
# Aumentar limite de tempo
python run_benchmark.py ... --timeout 60
```

### "Chrome not found" (Selenium)
```bash
# Instalar dependência
sudo apt-get install chromium-browser
# ou (macOS)
brew install chromium
```

### "Out of memory" (Selenium)
```bash
# Reduzir URLs e usar --concurrency 1
python run_benchmark.py ... --limit 30 --concurrency 1
```

---

## 📊 Resultado Final

Ao fim da execução completa, você terá:

```
results_standard/
├── results_<timestamp>.csv      # Dados brutos (todos os downloads)
├── REPORT.md                    # Relatório em Markdown
├── benchmark_<timestamp>.log    # Log detalhado
└── plots/
    ├── wall_time_boxplot.png    # Gráfico de latências
    ├── throughput_bar.png       # Produção por abordagem
    ├── memory_bar.png           # Uso de memória
    └── success_rate.png         # Taxa de sucesso %
```

**Para visualizar:**
```bash
cat results_standard/REPORT.md
# ou
open results_standard/REPORT.md  # macOS
xdg-open results_standard/REPORT.md  # Linux
```

---

## 📚 Documentação Completa

Para mais detalhes, ver [README.md](README.md)

- Estrutura dos arquivos
- Argumentos detalhados
- Customização
- Formato de dados
- API de classes

---

## ✅ Checklist

Antes de executar:

- [ ] Python 3.8+: `python --version`
- [ ] Dependências: `pip list | grep -E "requests|aiohttp|selenium|psutil"`
- [ ] Dataset: `ls ../../../dados/scraping_results/scraping_*.json`
- [ ] Espaço em disco: `df -h` (recomendado > 1GB)
- [ ] Conexão: `curl -I https://www.google.com`

Após execução:

- [ ] CSV gerado: `ls results_*/results_*.csv`
- [ ] Relatório: `cat results_*/REPORT.md`
- [ ] Gráficos: `ls results_*/plots/*.png`

---

## 🚀 Próximos Passos

1. **Executar teste rápido:**
   ```bash
   ./quick_test.sh 50
   ```

2. **Revisar relatório:**
   ```bash
   cat bench_results_quick/report.md
   ```

3. **Ajustar parâmetros conforme necessidade**

4. **Executar teste padrão completo:**
   ```bash
   python run_benchmark.py --dataset ... --output results_production
   ```

---

Dúvidas? Ver `validate_env.py`, `README.md` ou verificar logs em `bench_results/benchmark_*.log`

**Boa sorte! 🎯**
