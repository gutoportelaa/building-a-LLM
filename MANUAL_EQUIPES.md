# MANUAL DE EQUIPES — Extração de Texto DOM-PI
### Diário Oficial dos Municípios do Piauí · Pipeline de Corpus para LLM

> **Escopo:** Os PDFs já foram coletados. Este manual instrui **exclusivamente** a etapa de extração de texto estruturado e a consolidação dos datasets de todos os 13 territórios em um corpus unificado.

---

## Índice

1. [Visão Geral do Fluxo](#1-visão-geral-do-fluxo)
2. [Territórios e Slugs Padronizados](#2-territórios-e-slugs-padronizados)
3. [Pré-requisitos e Instalação](#3-pré-requisitos-e-instalação)
4. [Passo 1 — Criar a Estrutura de Diretórios](#4-passo-1--criar-a-estrutura-de-diretórios)
5. [Passo 2 — Depositar os PDFs (Drop-Zone)](#5-passo-2--depositar-os-pdfs-drop-zone)
6. [Passo 3 — Executar a Extração](#6-passo-3--executar-a-extração)
7. [Passo 4 — Verificar a Saída](#7-passo-4--verificar-a-saída)
8. [Schema de Dados Unificado](#8-schema-de-dados-unificado)
9. [Referência de Todos os Parâmetros](#9-referência-de-todos-os-parâmetros)
10. [Solução de Problemas](#10-solução-de-problemas)

---

## 1. Visão Geral do Fluxo

```
[PDFs coletados]
       │
       ▼
territorios/<slug>/pdfs/   ← Equipe deposita os PDFs aqui
       │
       ▼  uv run python src/dompi_scraper/extrair_territorio.py --territorio <slug>
       │
       ├─► extraidos/<slug>/datalake/    ← Arquivos .md com frontmatter YAML
       ├─► extraidos/<slug>/corpus_<slug>.jsonl  ← JSONL para fine-tuning
       └─► logs/<slug>/extracao_YYYYMMDD.log
```

**Um único comando por território.** O script:
1. Escaneia os PDFs na drop-zone e calcula SHA-256 de cada um
2. Avalia qualidade de cada PDF (PyMuPDF, sem GPU)
3. PDFs digitais nativos → extração rápida via **PyMuPDF**
4. PDFs escaneados → extração estruturada via **Marker na GPU**
5. Grava resultados no Data Lake e no JSONL

---

## 2. Territórios e Slugs Padronizados

> **Regra:** Use sempre o `slug` na linha de comando e o `Nome Canônico` no campo `territorio` do JSONL.

| # | Slug (usar no comando) | Nome Canônico (usar no JSONL) |
|---|---|---|
| 1 | `planice_litoran` | `Planície Litorânea` |
| 2 | `cocais` | `Cocais` |
| 3 | `carnaubais` | `Carnaubais` |
| 4 | `entre_rios` | `Entre Rios` |
| 5 | `vale_do_sambito` | `Vale do Sambito` |
| 6 | `vale_do_rio_guaribas` | `Vale do Rio Guaribas` |
| 7 | `chapada_vale_do_rio_itaim` | `Chapada Vale do Rio Itaim` |
| 8 | `vale_do_caninde` | `Vale do Canindé` |
| 9 | `serra_da_capivara` | `Serra da Capivara` |
| 10 | `vale_dos_rios_piaui_e_itaueiras` | `Vale dos Rios Piauí e Itaueiras` |
| 11 | `tabuleiros_alto_parnaiba` | `Tabuleiros do Alto Parnaíba e Chapada das Mangabeiras` |
| 12 | `teresina` | `Teresina` |
| 13 | `parnaiba` | `Parnaíba` |

---

## 3. Pré-requisitos e Instalação

### 3.1 Dependências de sistema

```bash
sudo apt update && sudo apt install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    libgl1-mesa-glx

# Confirme o Tesseract
tesseract --version
tesseract --list-langs | grep por
```

### 3.2 Clone e ambiente Python

```bash
# Clone o repositório (se ainda não fez)
git clone <URL_DO_REPOSITÓRIO> building-a-LLM
cd building-a-LLM

# Instala o gerenciador uv (se não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # ou reinicie o terminal

# Instala todas as dependências do projeto
uv sync

# Confirme que o ambiente está funcionando
uv run python -c "import fitz, torch; print('OK')"
```

### 3.3 Verificar GPU

```bash
uv run python -c "
import torch
print('CUDA disponível:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM total: {vram:.1f} GB')
else:
    print('GPU não disponível — use --force-ocr=False (PyMuPDF) ou Tesseract')
"
```

> **Com GPU (RTX 4070 Laptop):**
> ```
> CUDA disponível: True
> GPU: NVIDIA GeForce RTX 4070 Laptop GPU
> VRAM total: 8.0 GB
> ```
> **Sem GPU:** o script continua funcionando, usando apenas PyMuPDF para documentos nativos. PDFs escaneados exigirão `--sem-gpu` (veja Seção 6.3).

---

## 4. Passo 1 — Criar a Estrutura de Diretórios

Execute **uma única vez** na raiz do projeto:

```bash
bash setup_territorios.sh
```

Isso criará a seguinte estrutura completa:

```
building-a-LLM/
│
├── territorios/                   ← DROP-ZONE por território
│   ├── planice_litoran/
│   │   └── pdfs/                 ← Equipe 1 deposita PDFs aqui
│   ├── cocais/
│   │   └── pdfs/
│   ├── carnaubais/
│   │   └── pdfs/
│   ├── entre_rios/
│   │   └── pdfs/
│   ├── vale_do_sambito/
│   │   └── pdfs/
│   ├── vale_do_rio_guaribas/
│   │   └── pdfs/
│   ├── chapada_vale_do_rio_itaim/
│   │   └── pdfs/
│   ├── vale_do_caninde/
│   │   └── pdfs/
│   ├── serra_da_capivara/
│   │   └── pdfs/
│   ├── vale_dos_rios_piaui_e_itaueiras/
│   │   └── pdfs/
│   ├── tabuleiros_alto_parnaiba/
│   │   └── pdfs/
│   ├── teresina/
│   │   └── pdfs/
│   └── parnaiba/
│       └── pdfs/
│
├── extraidos/                     ← Saída da extração (gerado automaticamente)
│   └── <slug>/
│       ├── datalake/              ← Arquivos .md particionados
│       ├── corpus_<slug>.jsonl    ← JSONL para fine-tuning
│       ├── corpus_<slug>.meta.jsonl
│       ├── download_manifest.json ← Gerado automaticamente
│       └── registro_dedup_marker.json
│
├── logs/                          ← Logs de cada execução
│   └── <slug>/
│       └── extracao_YYYYMMDD_HHMMSS.log
│
└── corpus_final/                  ← Corpus unificado de todos os territórios
    ├── corpus_unificado.jsonl
    └── relatorio_consolidacao.json
```

---

## 5. Passo 2 — Depositar os PDFs (Drop-Zone)

Cada equipe responsável por um território deve copiar seus PDFs para a pasta correspondente:

```bash
# Exemplo: equipe do território Carnaubais
cp /caminho/dos/pdfs/coletados/*.pdf territorios/carnaubais/pdfs/

# Exemplo: equipe de Teresina (usando rsync para pastas grandes)
rsync -avh /origem/pdfs_teresina/ territorios/teresina/pdfs/

# Verificar quantos PDFs foram depositados
ls territorios/carnaubais/pdfs/ | wc -l
```

> **Regras para os PDFs:**
> - Apenas arquivos `.pdf` são processados (outros arquivos são ignorados)
> - Nomes de arquivo não importam — o sistema usa SHA-256 para identificação
> - Não é necessário renomear ou organizar internamente

---

## 6. Passo 3 — Executar a Extração

### 6.1 Teste Rápido (3 PDFs, qualquer território)

Sempre faça um teste antes da extração completa:

```bash
uv run python src/dompi_scraper/extrair_territorio.py \
    --territorio carnaubais \
    --limite 3 \
    --verbose
```

### 6.2 Extração Completa — Com GPU (Recomendado)

```bash
# Planície Litorânea
uv run python src/dompi_scraper/extrair_territorio.py --territorio planice_litoran

# Cocais
uv run python src/dompi_scraper/extrair_territorio.py --territorio cocais

# Carnaubais
uv run python src/dompi_scraper/extrair_territorio.py --territorio carnaubais

# Entre Rios
uv run python src/dompi_scraper/extrair_territorio.py --territorio entre_rios

# Vale do Sambito
uv run python src/dompi_scraper/extrair_territorio.py --territorio vale_do_sambito

# Vale do Rio Guaribas
uv run python src/dompi_scraper/extrair_territorio.py --territorio vale_do_rio_guaribas

# Chapada Vale do Rio Itaim
uv run python src/dompi_scraper/extrair_territorio.py --territorio chapada_vale_do_rio_itaim

# Vale do Canindé
uv run python src/dompi_scraper/extrair_territorio.py --territorio vale_do_caninde

# Serra da Capivara
uv run python src/dompi_scraper/extrair_territorio.py --territorio serra_da_capivara

# Vale dos Rios Piauí e Itaueiras
uv run python src/dompi_scraper/extrair_territorio.py --territorio vale_dos_rios_piaui_e_itaueiras

# Tabuleiros do Alto Parnaíba e Chapada das Mangabeiras
uv run python src/dompi_scraper/extrair_territorio.py --territorio tabuleiros_alto_parnaiba

# Teresina
uv run python src/dompi_scraper/extrair_territorio.py --territorio teresina

# Parnaíba
uv run python src/dompi_scraper/extrair_territorio.py --territorio parnaiba
```

### 6.3 Extração Sem GPU (CPU — Mais Lento)

Se a máquina não tiver GPU, o orquestrador usará apenas PyMuPDF. Para PDFs escaneados, ajuste o threshold para que nenhum documento seja enviado ao Marker:

```bash
# --threshold 1.1 → nenhum PDF vai para o Marker (força PyMuPDF para tudo)
uv run python src/dompi_scraper/extrair_territorio.py \
    --territorio carnaubais \
    --threshold 1.1
```

> **Qualidade:** PDFs escaneados extraídos só com PyMuPDF podem ter qualidade inferior. Recomenda-se marcar `engine_extracao: "PyMuPDF-CPU"` manualmente caso precise diferenciar no corpus.

### 6.4 Force-OCR para PDFs Escaneados em Massa

Quando você sabe que o lote é 100% escaneado e quer garantir máxima qualidade de tabelas:

```bash
uv run python src/dompi_scraper/extrair_territorio.py \
    --territorio vale_do_sambito \
    --force-ocr
```

> **Atenção:** `--force-ocr` usa ~3.2 GB de VRAM. Não combine com outros processos GPU.

### 6.5 Extração Incremental (Retomada após Interrupção)

O script é **idempotente**: re-executar o mesmo comando pula PDFs já processados (verificados via registro de deduplicação).

```bash
# Se a extração foi interrompida, basta executar o mesmo comando novamente
uv run python src/dompi_scraper/extrair_territorio.py --territorio carnaubais
# → PDFs já extraídos são pulados automaticamente
```

---

## 7. Passo 4 — Verificar a Saída

### 7.1 Conferir Arquivos Gerados

```bash
# Quantos documentos foram extraídos
find extraidos/carnaubais/datalake -name "*.md" | wc -l

# Ver o JSONL gerado
head -n 1 extraidos/carnaubais/corpus_carnaubais.jsonl | python -m json.tool | head -30

# Ver log da extração
cat logs/carnaubais/$(ls -t logs/carnaubais/ | head -1)
```

### 7.2 Validar Schema JSONL

```bash
uv run python -c "
import json, sys

campos = ['doc_id','territorio','municipio','data_publicacao','texto_markdown','engine_extracao']
slug = 'carnaubais'  # <- altere para o seu território

erros = 0
total = 0
with open(f'extraidos/{slug}/corpus_{slug}.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        total += 1
        try:
            doc = json.loads(line)
            for c in campos:
                if c not in doc or not doc[c]:
                    print(f'Linha {i}: campo obrigatório ausente ou vazio: {c}')
                    erros += 1
        except json.JSONDecodeError as e:
            print(f'Linha {i}: JSON inválido — {e}')
            erros += 1

print(f'Total: {total} | Erros: {erros}')
if erros == 0:
    print('✅ Validação OK — JSONL pronto para consolidação')
"
```

### 7.3 Resumo por Território

```bash
# Conta documentos extraídos em todos os territórios
for slug in planice_litoran cocais carnaubais entre_rios vale_do_sambito \
            vale_do_rio_guaribas chapada_vale_do_rio_itaim vale_do_caninde \
            serra_da_capivara vale_dos_rios_piaui_e_itaueiras \
            tabuleiros_alto_parnaiba teresina parnaiba; do
    count=$(find "extraidos/${slug}/datalake" -name "*.md" 2>/dev/null | wc -l)
    printf "  %-40s %s documentos\n" "${slug}" "${count}"
done
```

---

## 8. Schema de Dados Unificado

Cada linha do arquivo `.jsonl` gerado segue este formato obrigatório:

### 8.1 Campos

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `doc_id` | String | ✅ | SHA-256 do arquivo PDF físico |
| `territorio` | String | ✅ | Nome canônico da Tabela em §2 |
| `municipio` | String | ✅ | Grafia IBGE exata (ex: `Assunção do Piauí`) |
| `data_publicacao` | String | ✅ | ISO 8601 (`YYYY-MM-DD`) ou `null` |
| `texto_markdown` | String | ✅ | Conteúdo extraído com `#`, `##`, tabelas Markdown |
| `metadados_extraidos` | Objeto | ⬜ | Entidades extras: CNPJs, número de licitação, etc. |
| `engine_extracao` | String | ✅ | Motor usado (veja tabela abaixo) |

### 8.2 Valores de `engine_extracao`

| Valor | Quando |
|---|---|
| `"Orquestrador-Marker"` | PDF escaneado → Marker GPU (padrão slow path) |
| `"Orquestrador-PyMuPDF"` | PDF nativo → PyMuPDF (padrão fast path) |
| `"Marker"` | `extrator_marker.py` puro |
| `"Tesseract"` | `orquestrador_tesseract.py` |
| `"PyMuPDF"` | `processar_pdfs.py` puro |

### 8.3 Exemplo de Registro

```json
{
  "doc_id": "e3b0c44298fc1c149afbf4c8996fb924...b855",
  "territorio": "Vale do Rio Guaribas",
  "municipio": "Campo Maior",
  "data_publicacao": "2025-03-15",
  "texto_markdown": "# PREFEITURA MUNICIPAL DE CAMPO MAIOR\n\n## PORTARIA Nº 042/2025\n\n**RESOLVE:**\n\n| Servidor | Cargo |\n|---|---|\n| João Silva | Agente Administrativo |",
  "metadados_extraidos": {
    "tipo_ato": "Portaria",
    "num_ato": "042/2025",
    "entidade": "Prefeitura Municipal",
    "edicao": "5231"
  },
  "engine_extracao": "Orquestrador-Marker"
}
```

### 8.4 Grafia IBGE dos Municípios

Sempre use a grafia oficial do IBGE no campo `municipio`:

| ❌ Errado | ✅ Correto (IBGE) |
|---|---|
| `campo_maior` | `Campo Maior` |
| `Assuncao do Piaui` | `Assunção do Piauí` |
| `Sao Joao da Serra` | `São João da Serra` |
| `Cabeçeiras` | `Cabeceiras do Piauí` |
| `Teresina/PI` | `Teresina` |
| `Parnaiba` | `Parnaíba` |

---

## 9. Referência de Todos os Parâmetros

```
uv run python src/dompi_scraper/extrair_territorio.py [opções]
```

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--territorio SLUG` | — | **Obrigatório.** Slug do território (veja §2) |
| `--limite N` | ilimitado | Processa no máximo N PDFs. Use `3` para testes |
| `--threshold 0.0–1.0` | `0.70` | Score OCR: acima → PyMuPDF, abaixo → Marker |
| `--force-ocr` | off | Força Marker com OCR completo em todos os PDFs |
| `--min-variance N` | `50.0` | Nitidez mínima (Laplacian). Abaixo = PDF em branco/corrompido |
| `--verbose` | off | Ativa logs DEBUG detalhados |
| `--listar` | — | Lista todos os territórios e slugs disponíveis |

**Exemplos rápidos:**

```bash
# Ver todos os territórios disponíveis
uv run python src/dompi_scraper/extrair_territorio.py --listar

# Teste com 3 PDFs e log detalhado
uv run python src/dompi_scraper/extrair_territorio.py --territorio parnaiba --limite 3 --verbose

# Produção padrão (Orquestrador Híbrido com GPU)
uv run python src/dompi_scraper/extrair_territorio.py --territorio vale_do_sambito

# Forçar OCR completo (máxima qualidade, mais lento)
uv run python src/dompi_scraper/extrair_territorio.py --territorio cocais --force-ocr

# Sem GPU (apenas PyMuPDF)
uv run python src/dompi_scraper/extrair_territorio.py --territorio entre_rios --threshold 1.1
```

---

## 10. Solução de Problemas

### ❌ `Pasta não encontrada: territorios/<slug>/pdfs`

Execute primeiro o script de setup:
```bash
bash setup_territorios.sh
```

### ❌ `Nenhum PDF encontrado em territorios/<slug>/pdfs`

Copie os PDFs para a pasta correta:
```bash
cp /origem/*.pdf territorios/<slug>/pdfs/
ls territorios/<slug>/pdfs/ | wc -l  # confirme que aparecem
```

### ❌ `CUDA disponível: False` (GPU não detectada)

```bash
# Verifique o driver
nvidia-smi

# Se o driver estiver OK mas o torch não vê a GPU, reinstale com CUDA:
uv add "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121
```

### ❌ `ImportError: No module named 'marker'`

```bash
uv sync
# Se persistir:
uv add marker-pdf
```

### ❌ `tesseract: command not found`

```bash
sudo apt install tesseract-ocr tesseract-ocr-por
```

### ❌ `CUDA out of memory`

A GPU ficou sem memória. Soluções:

```bash
# 1. Verifique quem está usando a GPU
nvidia-smi

# 2. Aumente o limiar de variância para pular PDFs muito pesados
uv run python src/dompi_scraper/extrair_territorio.py \
    --territorio carnaubais \
    --min-variance 100.0

# 3. Processe em lotes menores
uv run python src/dompi_scraper/extrair_territorio.py \
    --territorio carnaubais \
    --limite 50
# → Depois re-execute (o script retoma do ponto onde parou)
```

### ❌ `texto_markdown` está vazio ou com lixo (OCR ruim)

```bash
# Reprocesse com force-ocr (Marker fará OCR do zero)
uv run python src/dompi_scraper/extrair_territorio.py \
    --territorio <slug> \
    --force-ocr \
    --min-variance 20.0   # mais permissivo para aceitar PDFs ruins
```

### ❌ Município aparece como `DESCONHECIDO`

O PDF não tem cabeçalho `PREFEITURA/CÂMARA DE <Cidade>` legível. O registro ainda é gerado com o nome do território como fallback. Verifique e corrija manualmente no JSONL se necessário.

---

## Arquivos de Referência

| Arquivo | Descrição |
|---|---|
| [`setup_territorios.sh`](setup_territorios.sh) | Cria toda a estrutura de diretórios |
| [`src/dompi_scraper/extrair_territorio.py`](src/dompi_scraper/extrair_territorio.py) | **Script principal de extração por território** |
| [`src/dompi_scraper/orquestrador_extracao.py`](src/dompi_scraper/orquestrador_extracao.py) | Orquestrador Híbrido (PyMuPDF + Marker) |
| [`src/dompi_scraper/extrator_marker.py`](src/dompi_scraper/extrator_marker.py) | Motor Marker puro (GPU) |
| [`src/dompi_scraper/orquestrador_tesseract.py`](src/dompi_scraper/orquestrador_tesseract.py) | Motor Tesseract (CPU fallback) |
| [`CONTEXT.md`](CONTEXT.md) | Arquitetura completa do pipeline DOM-PI |
| [`CONTEXT_2.md`](CONTEXT_2.md) | Relatório de triagem VLM/OCR |

---

*Última atualização: 2026-05-26 — DOM-PI Pipeline v0.1 · 13 territórios*
