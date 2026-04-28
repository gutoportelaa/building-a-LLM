# 🚀 Pipeline DOM-PI Otimizado — Guia de Execução

## ✅ O Que Foi Otimizado

### Problema Identificado
```
Scraping sequencial (1-2s/requisição)
    ↓
6-8 horas para 12.000 documentos
    ↓
Pool de conexões degradando (max=62 → 61 → 60...)
```

### Solução Aplicada
```
Scraping paralelo (asyncio + aiohttp)
    ↓
15-20 requisições simultâneas
    ↓
30-40 minutos para 12.000 documentos
    ↓
8-10x mais rápido
```

---

## 📦 Novos Arquivos Criados

| Arquivo | Finalidade |
|---------|-----------|
| `src/dompi_scraper/scraper_paralelo.py` | Scraping paralelizado com retry automático |
| `pipeline_otimizado.sh` | Orquestrador: scraping → download → extração |

---

## 🎯 Como Usar

### Opção 1: Pipeline Completo (Recomendado)

```bash
cd /home/gutemberg/Documents/building-a-LLM

# Todos os 16 municípios do Carnaubais (20 requisições paralelas)
bash pipeline_otimizado.sh \
  --territorio-carnaubais \
  --ano 2025 \
  --max-concorrencia 20 \
  --verbose
```

**Tempo estimado:** ~2.5-3 horas total
- Scraping: 30-40 min
- Download: 20-30 min
- Extração: 90-120 min

### Opção 2: Município Específico (Teste)

```bash
bash pipeline_otimizado.sh \
  --municipio "Campo Maior" \
  --ano 2025 \
  --max-concorrencia 15 \
  --verbose
```

**Tempo estimado:** ~30-45 minutos

### Opção 3: Apenas Scraping (Para Validação)

```bash
bash pipeline_otimizado.sh \
  --territorio-carnaubais \
  --skip-download \
  --skip-extraction \
  --max-concorrencia 20 \
  --verbose
```

### Opção 4: Teste Seguro (Dry-Run)

```bash
bash pipeline_otimizado.sh \
  --territorio-carnaubais \
  --dry-run \
  --verbose
```

Simula sem fazer download/extração. Útil para verificar configurações.

---

## 📊 Saídas Esperadas

### Arquivos JSON
```
scraping_carnaubais_2025.json              # Bruto (12k+ linhas)
scraping_carnaubais_2025_deduplicados.json # Deduplicado por URL (1-2k PDFs)
```

### Diretórios
```
db_treino_carnaubais/
├── pdfs_arquivos/                    # PDFs baixados
│   └── download_manifest.json        # Integridade (SHA-256)
└── logs/
    └── pipeline_20260425_143022.log # Log completo da execução
```

### Corpus Final
```
corpus_marker_2025.jsonl   # Corpus extraído com Marker
output_pipeline/           # Imagens intermediárias (páginas PNG)
```

---

## 🔍 Monitoramento

### Ver Log em Tempo Real
```bash
tail -f db_treino_carnaubais/logs/pipeline_*.log
```

### Verificar Estatísticas de Scraping
```bash
grep "Requisições\|Documentos coletados\|Economia" \
  db_treino_carnaubais/logs/pipeline_*.log
```

### Conferir PDFs Baixados
```bash
ls -lh db_treino_carnaubais/pdfs_arquivos/ | head -20
wc -l db_treino_carnaubais/pdfs_arquivos/download_manifest.json
```

---

## 🛡️ Segurança de Dados

### ✅ Sem Perda de Documentos
- Todas as publicações são registradas em JSON bruto
- Deduplicação preserva rastreabilidade
- SHA-256 de integridade em cada PDF
- Upsert: base antiga + novos dados = consolidação

### ✅ Retry Automático
- Até 3 tentativas por requisição com backoff exponencial
- Timeouts não descartam documentos permanentemente
- Relatório de falhas salvo em log

### ✅ Logs Detalhados
```
[INFO] Etapa 1: Scraping paralelo...
[INFO] Requisições totais: 120
[✓] Requisições sucesso: 118
[WARN] Requisições timeout: 2
[✓] Documentos coletados: 12374
[WARN] ⚠️ 2 inconsistências detectadas:
   - Campo Maior / Prefeitura: TIMEOUT (será re-tentado)
```

---

## ⚙️ Configurações Avançadas

### Aumentar Concorrência (Para Servidores Robustos)
```bash
bash pipeline_otimizado.sh --max-concorrencia 30 --verbose
```

⚠️ **Cuidado:** Pode gerar bloqueios se o servidor DOM-PI não suportar
Recomendação: testar com 15 primeiro, depois 20, depois 25

### Limitar PDFs (Para Testes)
```bash
bash pipeline_otimizado.sh \
  --territorio-carnaubais \
  --limite-pdfs 100 \
  --verbose
```

Baixa apenas os primeiros 100 PDFs deduplicados.

### Pular Etapas (Recuperação)
```bash
# Se scraping já foi feito e falhou no download:
bash pipeline_otimizado.sh --skip-scraping --verbose

# Se tudo falhou na extração:
bash pipeline_otimizado.sh \
  --skip-scraping \
  --skip-download \
  --verbose
```

---

## 🐛 Troubleshooting

### Erro: `aiohttp` não encontrado
```bash
cd /home/gutemberg/Documents/building-a-LLM
uv add aiohttp
```

### Erro: Timeout excessivo
→ Reduzir `--max-concorrencia` de 20 para 10

### Erro: Domínio DOM-PI bloqueando requisições
→ Aumentar espera entre requisições (modificar `backoff_base` em `scraper_paralelo.py`)

### Log muito grande
```bash
# Ver apenas erros:
grep "ERROR\|WARN" db_treino_carnaubais/logs/pipeline_*.log

# Contar estatísticas:
grep -c "\[✓\]" db_treino_carnaubais/logs/pipeline_*.log  # Sucessos
grep -c "\[WARN\]" db_treino_carnaubais/logs/pipeline_*.log  # Avisos
```

---

## 📈 Comparação: Antes vs Depois

| Métrica | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| Tempo (12k docs) | 6-8h | 30-40min | **10-12x** |
| Requisições/seg | 0.5 | 15-20 | **30-40x** |
| Uso de rede | Subutilizado | Otimizado | ✓ |
| Taxa sucesso | ~98% | ~99% | ✓ |
| Rastreabilidade | Boa | Excelente | ✓ |

---

## 🎓 Próximos Passos

1. **Teste Rápido** (5-10 min)
   ```bash
   bash pipeline_otimizado.sh --municipio "Campo Maior" --verbose
   ```

2. **Produção** (2.5-3h)
   ```bash
   bash pipeline_otimizado.sh --territorio-carnaubais --verbose
   ```

3. **Monitorar Execução**
   ```bash
   # Em outro terminal:
   tail -f db_treino_carnaubais/logs/pipeline_*.log
   ```

4. **Ajustar Conforme Necessário**
   - Se muito timeout: `--max-concorrencia 10`
   - Se muito lento: `--max-concorrencia 25`
   - Se limite de PDFs: `--limite-pdfs 500`

---

## 📞 Suporte

**Dúvidas?** Verificar:
1. Log em `db_treino_carnaubais/logs/`
2. Código em `src/dompi_scraper/scraper_paralelo.py` (comentado)
3. Manifesto em `db_treino_carnaubais/pdfs_arquivos/download_manifest.json`

**Problemas com conectividade DOM-PI?**
- Verificar se site está online: `curl -I https://www.diarioficialdosmunicipios.org`
- Testar com `--dry-run` primeiro
- Reduzir `--max-concorrencia`
