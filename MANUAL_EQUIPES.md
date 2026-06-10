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
       ▼  PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio <slug>
       │
       ├─► extraidos/<slug>/datalake/          ← Arquivos .md com frontmatter YAML
       ├─► extraidos/<slug>/corpus_<slug>.jsonl ← JSONL para fine-tuning
       └─► logs/<slug>/extracao_YYYYMMDD.log
```

**Um único comando por território.** O script:

1. Escaneia os PDFs na drop-zone e calcula SHA-256 de cada um
2. Executa triagem DLA (Document Layout Analysis) via **PyMuPDF** — mapeia páginas por município, calcula score OCR e detecta complexidade (tabelas, keywords fiscais)
3. Roteia cada chunk de município para o extrator adequado ao hardware:

A **rota é decidida primariamente pelo NOME do arquivo** (sinal confiável e gratuito):
documentos da família **licitação / contrato / relatório fiscal** (`Licitacao`, `Pregao`,
`Contrato`, `Extrato`, `LRF`, `RREO`, `RGF`, `Anexo`, ...) têm tabelas/valores → **Docling**;
os **demais** (`Portaria`, `Decreto`, `Lei`, `Aviso`, ...) são texto comum. A detecção
estrutural de tabela no conteúdo (`find_tables` / densidade de valores monetários) pode
**promover** um documento "comum" para a rota Docling.

| Tipo de documento | Com GPU | Sem GPU (modesto) |
|---|---|---|
| **Comum** nativo (portaria, decreto, lei...) | PaddleOCR CUDA | PyMuPDF (instantâneo) |
| **Fiscal/licitação/contrato** (tabelas/valores) | Docling CUDA | Docling CPU (`do_ocr=False`) |
| **Escaneado** (sem texto nativo) | PaddleOCR CUDA | Tesseract (PT-BR) |

> **Validação prévia da rota (recomendado):** rode `--dry-run-rota` para classificar todos os
> documentos (nome → fiscal/comum + detecção de tabela) e gerar `relatorio_rota.ndjson`, **sem
> rodar nenhum motor** — assim a equipe audita a separação antes de gastar GPU.

> **Datas:** a análise cronológica é feita **somente pelo padrão do nome do arquivo**
> (`extrair_data_filename`), nunca pelo texto — datas de vigência/referência no corpo
> contaminavam a partição. A partição usa `ano=<AAAA>/mes=sem_mes` (mês/dia exige o
> mapeamento edição→data, ainda pendente).

4. Manipula e grava resultados com **Polars** (NDJSON/JSONL) no Data Lake

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

# Monta os DOIS ambientes isolados (torch/docling em .venv; paddleocr em .venv-paddle).
# Idempotente: rode quantas vezes quiser. Em WSL/1-GPU o build CPU do paddle basta.
bash setup_venvs.sh

# Para a máquina-lab com GPU dedicada ao PaddleOCR (build cu126):
#   bash setup_venvs.sh --paddle-gpu
```

> ⚠️ **REGRA DE OURO — nunca use `uv run` para extrair.** O `uv run`/`uv sync` re-sincroniza
> o `.venv` e pode reinstalar o paddle, **quebrando o `import torch`** (conflito de libs
> `nvidia/` entre `paddlepaddle-gpu` cu126 e `torch` cu13). Invoque o interpretador
> direto: `PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio ...`.
> O orquestrador dispara o `.venv-paddle` sozinho, em subprocesso isolado.

> **Por que dois venvs?** `torch` (Docling) e `paddlepaddle-gpu` (PaddleOCR) sobrescrevem
> as mesmas libs CUDA e não coexistem no mesmo ambiente. `setup_venvs.sh` cria:
> `.venv` (torch + docling + orquestrador) e `.venv-paddle` (paddleocr isolado).

### 3.3 Verificar GPU

```bash
./.venv/bin/python -c "
import torch
print('CUDA disponível:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM total: {vram:.1f} GB')
else:
    print('GPU não disponível — usando Tesseract + PaddleOCR CPU automaticamente')
"
```

> **Com GPU (ex: RTX 4070 Laptop):**
> ```
> CUDA disponível: True
> GPU: NVIDIA GeForce RTX 4070 Laptop GPU
> VRAM total: 8.0 GB
> ```
> **Sem GPU:** o script detecta automaticamente e usa Tesseract para OCR e PaddleOCR CPU para tabelas. Nenhuma configuração manual necessária.

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
│       ├── corpus_<slug>.jsonl    ← JSONL para fine-tuning (Polars NDJSON)
│       ├── corpus_<slug>.meta.jsonl
│       ├── download_manifest.json ← Gerado automaticamente
│       └── registro_dedup.ndjson  ← Dedup persistente (Polars)
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

Cada equipe responsável por um território deve copiar seus PDFs para a pasta correspondente. O script suporta que os arquivos estejam soltos na pasta `pdfs/` ou organizados em subpastas.

Se você organizar os PDFs em subpastas com o nome do município e da entidade, o script **automaticamente inferirá** esses dados para o seu JSONL final:

```bash
# Estrutura recomendada para facilitar a inferência automática:
territorios/tabuleiros_alto_parnaiba/pdfs/
├── Marcos_Parente/
│   ├── Prefeitura/
│   │   └── decreto_01.pdf
│   └── Camara/
│       └── portaria_02.pdf
└── Urucui/
    └── arquivo_avulso.pdf
```

Nesse caso, `decreto_01.pdf` receberá `municipio="Marcos Parente"` e `entidade="Prefeitura"` automaticamente.

Comandos úteis para copiar os arquivos:

```bash
# Exemplo 1: Copiar todos os PDFs já organizados (mantendo as pastas)
cp -r /origem/pdfs_cidade/* territorios/carnaubais/pdfs/

# Exemplo 2: Usar rsync para transferência segura de grandes volumes
rsync -avh /origem/pdfs_teresina/ territorios/teresina/pdfs/

# Verificar quantos PDFs foram depositados (incluindo subpastas)
find territorios/carnaubais/pdfs/ -type f -name "*.pdf" | wc -l
```

> **Regras para os PDFs:**
> - Apenas arquivos `.pdf` são processados (outros arquivos são ignorados).
> - O script faz busca recursiva, então não há limite para a profundidade das pastas.
> - Se não houver subpastas com o nome do município, o script usará o nome do território como fallback no campo `municipio` (o Orquestrador refinará o município pelo cabeçalho interno do documento).
> - Nomes de arquivo não importam — o sistema usa o hash SHA-256 para identificação única.

---

## 6. Passo 3 — Executar a Extração

### 6.1 Teste Rápido (3 PDFs, qualquer território)

Sempre faça um teste antes da extração completa:

```bash
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio carnaubais \
    --limite 3 \
    --verbose
```

### 6.2 Extração Completa

O script detecta GPU automaticamente — **o mesmo comando funciona com ou sem GPU**:

```bash
# Planície Litorânea
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio planice_litoran

# Cocais
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio cocais

# Carnaubais
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio carnaubais

# Entre Rios
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio entre_rios

# Vale do Sambito
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio vale_do_sambito

# Vale do Rio Guaribas
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio vale_do_rio_guaribas

# Chapada Vale do Rio Itaim
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio chapada_vale_do_rio_itaim

# Vale do Canindé
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio vale_do_caninde

# Serra da Capivara
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio serra_da_capivara

# Vale dos Rios Piauí e Itaueiras
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio vale_dos_rios_piaui_e_itaueiras

# Tabuleiros do Alto Parnaíba e Chapada das Mangabeiras
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio tabuleiros_alto_parnaiba

# Teresina
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio teresina

# Parnaíba
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio parnaiba
```

### 6.3 Extração Sem GPU (CPU — Stack Automática)

O orquestrador detecta a GPU (via torch) e adapta a stack. Em máquinas sem GPU o roteamento é:

- **Digital nativo** (score ≥ `--threshold`, padrão 0.45) → PyMuPDF (milissegundos por página)
- **Escaneado mundano** (score baixo, sem tabelas) → Tesseract PT-BR
- **Complexo** (tabela estrutural / densidade de valores monetários) → PaddleOCR CPU

> **WSL / GPU única com PaddleOCR build CPU:** se o `.venv-paddle` foi montado sem `--paddle-gpu`,
> rode os escaneados/complexos com o paddle em CPU explicitamente:
> ```bash
> PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio carnaubais --gpu-paddle cpu
> ```
> Como PDFs escaneados são raros no DOM-PI (corpus majoritariamente nativo), o PyMuPDF/Docling
> resolve a maioria e o paddle é acionado pouco.

> **Cap anti-OOM do Docling:** `--docling-max-paginas N` (padrão 8) fatia documentos longos em
> lotes de N páginas antes de enviar ao Docling — controle decisivo de memória em edições grandes.

> **Expectativa de velocidade CPU:** ~3–10 s/pág para Tesseract, ~20–60 s/pág para PaddleOCR CPU. Volumes grandes levam horas — considere usar `--limite` para lotes menores e retomar depois (o script é idempotente).

### 6.4 Extração Incremental (Retomada após Interrupção)

O script é **idempotente**: re-executar o mesmo comando pula PDFs já processados (verificados via `registro_dedup.ndjson`).

```bash
# Se a extração foi interrompida, basta executar o mesmo comando novamente
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio carnaubais
# → PDFs já extraídos são pulados automaticamente
```

---

## 7. Passo 4 — Verificar a Saída

### 7.1 Conferir Arquivos Gerados

```bash
# Quantos documentos foram extraídos
find extraidos/carnaubais/datalake -name "*.md" | wc -l

# Ver o JSONL gerado (Polars NDJSON)
./.venv/bin/python -c "
import polars as pl
df = pl.read_ndjson('extraidos/carnaubais/corpus_carnaubais.jsonl')
print(df.head(2))
print(df.schema)
"

# Ver log da extração
cat logs/carnaubais/$(ls -t logs/carnaubais/ | head -1)
```

### 7.2 Validar Schema JSONL com Polars

```bash
./.venv/bin/python -c "
import polars as pl

slug = 'carnaubais'  # <- altere para o seu território
df = pl.read_ndjson(f'extraidos/{slug}/corpus_{slug}.jsonl')

campos = ['id_publicacao', 'municipio', 'tipo_ato', 'data_publicacao', 'extrator', 'texto']
for c in campos:
    vazios = df.filter(pl.col(c).is_null() | (pl.col(c) == '')).height
    if vazios:
        print(f'  AVISO: {vazios} registros com campo vazio: {c}')

print(f'Total: {df.height} | Extratores usados:')
print(df.group_by('extrator').len().sort('len', descending=True))
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

Cada linha do arquivo `.jsonl` gerado segue este formato (Polars NDJSON):

### 8.1 Campos

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `id_publicacao` | String | ✅ | MD5 do conteúdo textual extraído |
| `territorio` | String | ✅ | Nome canônico do território |
| `municipio` | String | ✅ | Grafia IBGE exata (ex: `Assunção do Piauí`) |
| `tipo_ato` | String | ✅ | Portaria, Decreto, Lei, Edital, etc. |
| `data_publicacao` | String | ✅ | Ano (`AAAA`) derivado do nome do arquivo, ou `""` |
| `extrator` | String | ✅ | Motor usado (veja tabela abaixo) |
| `texto` | String | ✅ | Conteúdo Markdown com `#`, `##`, tabelas |
| `n_chars` | Int | ✅ | Comprimento do campo `texto` |

### 8.2 Valores de `extrator`

| Valor | Quando |
|---|---|
| `"pymupdf"` | Comum nativo sem GPU (rota padrão CPU) |
| `"paddle-cuda"` | Comum nativo OU escaneado, com GPU CUDA |
| `"docling-cuda"` | Fiscal/licitação/tabela com GPU CUDA |
| `"docling-cpu"` | Fiscal/licitação/tabela sem GPU |
| `"tesseract"` | Escaneado sem GPU |
| `"paddle-cuda-fallback"` | Docling indisponível → PaddleOCR CUDA |
| `"pymupdf-fallback"` | Motor OCR vazio → PyMuPDF (texto nativo) |

### 8.3 Exemplo de Registro

```json
{
  "id_publicacao": "a3f2c9d1e4b07825...",
  "municipio": "Campo Maior",
  "tipo_ato": "Portaria",
  "data_publicacao": "2025-03-15",
  "extrator": "docling-cuda",
  "texto": "# PREFEITURA MUNICIPAL DE CAMPO MAIOR\n\n## PORTARIA Nº 042/2025\n\n**RESOLVE:**\n\n| Servidor | Cargo |\n|---|---|\n| João Silva | Agente Administrativo |",
  "n_chars": 312
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
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio [opções]
```

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `--territorio SLUG` | — | **Obrigatório.** Slug do território (veja §2) |
| `--modo` | `paddle` | `paddle` (padrão) ou `pymu` (mais rápido, sem layout avançado) |
| `--limite N` | ilimitado | Processa no máximo N PDFs. Use `3` para testes |
| `--verbose` | off | Ativa logs DEBUG detalhados |
| `--listar` | — | Lista todos os territórios e slugs disponíveis |
| `--workers N` | `cpu_count()-1` | [Modo paddle] Processos paralelos |
| `--threshold` | `0.70` | [Legado] Score OCR para roteamento (ignorado no modo paddle) |
| `--force-ocr` | off | [Legado] Ignorado. Mantido por compatibilidade |
| `--min-variance` | `50.0` | [Legado] Ignorado. Mantido por compatibilidade |

**Exemplos rápidos:**

```bash
# Ver todos os territórios disponíveis
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --listar

# Teste com 3 PDFs e log detalhado
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio parnaiba --limite 3 --verbose

# Produção padrão (detecta GPU automaticamente)
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio vale_do_sambito

# Modo PyMuPDF apenas (mais rápido, sem análise de tabelas)
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio entre_rios --modo pymu
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

### ❌ `ImportError: No module named 'polars'`

```bash
uv sync
# Se persistir:
uv add "polars>=1.0.0"
```

### ❌ `ImportError: No module named 'paddleocr'`

```bash
uv sync
# Se persistir:
uv add paddleocr paddlepaddle
```

### ❌ `ImportError: No module named 'docling'` (apenas com GPU)

Docling é opcional. Instale se quiser extração de tabelas de alta fidelidade com GPU:
```bash
uv sync --extra docling
# ou
uv add docling
```
Sem Docling, documentos complexos são processados automaticamente com PaddleOCR CUDA como fallback.

### ❌ `tesseract: command not found`

```bash
sudo apt install tesseract-ocr tesseract-ocr-por
```

### ❌ `CUDA out of memory`

A GPU ficou sem memória. Soluções:

```bash
# 1. Verifique quem está usando a GPU
nvidia-smi

# 2. Processe em lotes menores (o script retoma do ponto onde parou)
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio carnaubais \
    --limite 50
# → Re-execute até completar (idempotente)
```

### ❌ `texto` está vazio ou com lixo (OCR ruim)

Verifique qual extrator foi usado no registro (`extrator` no JSONL). Se for `tesseract`, tente forçar o modo paddle para esse território:

```bash
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio <slug> \
    --modo paddle
```

### ❌ Município aparece como `DESCONHECIDO`

O PDF não tem cabeçalho `PREFEITURA/CÂMARA DE <Cidade>` legível. O registro ainda é gerado com o nome do território como fallback. Verifique e corrija manualmente no JSONL se necessário.

---

## Arquivos de Referência

| Arquivo | Descrição |
|---|---|
| [`setup_territorios.sh`](setup_territorios.sh) | Cria toda a estrutura de diretórios |
| [`src/dompi_scraper/extrair_territorio.py`](src/dompi_scraper/extrair_territorio.py) | **Script principal de extração por território** |
| [`src/dompi_scraper/orquestrador_extracao.py`](src/dompi_scraper/orquestrador_extracao.py) | Orquestrador Híbrido (PyMuPDF + PaddleOCR/Docling/Tesseract) |
| [`src/dompi_scraper/extrator_paddle.py`](src/dompi_scraper/extrator_paddle.py) | Motor PaddleOCR PP-Structure |
| [`src/dompi_scraper/worker_docling.py`](src/dompi_scraper/worker_docling.py) | Motor Docling GPU / Ollama CPU |
| [`src/dompi_scraper/orquestrador_tesseract.py`](src/dompi_scraper/orquestrador_tesseract.py) | Motor Tesseract (referência legada) |
| [`CONTEXT.md`](CONTEXT.md) | Arquitetura completa do pipeline DOM-PI |
| [`CONTEXT_2.md`](CONTEXT_2.md) | Relatório de triagem VLM/OCR |

---

*Última atualização: 2026-05-30 — DOM-PI Pipeline v0.2 · stack sem Marker · Polars NDJSON*
