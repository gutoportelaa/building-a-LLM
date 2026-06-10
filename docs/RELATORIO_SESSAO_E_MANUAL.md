# Relatório da Sessão + Manual de Uso — Pipeline DOM-PI

> **Branch:** `feat-paddle-unificacao` (baseada em `feat-paddle`)
> **Data:** 2026-06-01
> **Escopo:** unificação do pipeline de extração, calibração de roteamento, deploy e
> validação end-to-end no cluster GPU (SLURM).

Este documento é **didático e autocontido**. Está organizado em:
1. [O que mudou nesta sessão](#1-o-que-mudou-nesta-sessão)
2. [Como o pipeline funciona agora](#2-como-o-pipeline-funciona-agora)
3. [Manual de uso (local e cluster)](#3-manual-de-uso)
4. [Verificação pelo terminal](#4-verificação-pelo-terminal)
5. [Como funcionam os jobs (SLURM)](#5-como-funcionam-os-jobs-slurm)
6. [Checklist de execução por território](#6-checklist-de-execução-por-território)
7. [Recursos disponíveis](#7-recursos-disponíveis)
8. [Possíveis aprimoramentos](#8-possíveis-aprimoramentos)

---

## 1. O que mudou nesta sessão

### 1.1 Unificação do pipeline (a mudança estrutural)
Antes existiam **duas pipelines divergentes**: o `extrair_territorio.py` rodava um caminho
PaddleOCR-CPU em `ProcessPoolExecutor` (propenso a OOM no WSL) e ignorava GPU; o
`orquestrador_extracao.py` (workers isolados, validado) ficava órfão. **Agora o
`extrair_territorio.py` delega ao orquestrador** e o caminho antigo foi aposentado.

| Arquivo | Mudança |
|---|---|
| `extrair_territorio.py` | Modo padrão delega ao orquestrador; `run_modo_paddle`/`_paddle_worker_task`/`calcular_limite_seguro_workers` **removidos**; novos args (`--threshold`, `--dpi`, `--docling-max-paginas`, `--gpu-paddle`, `--gpu-docling`, `--python-*`, `--dry-run-rota`); logs do orquestrador/workers unificados no `.log` do território (arquivo grava DEBUG sempre); correção do `--listar`; correção do `--gpu-paddle cpu` (não vira mais "auto"). |
| `orquestrador_extracao.py` | **Roteamento por nome de arquivo** (fiscal→Docling, comum→PaddleOCR/PyMuPDF); **datas só pelo nome**; **cap anti-OOM do Docling** (`--docling-max-paginas`, fatiamento); **correção do corpus-resume** (não sobrescreve o NDJSON na retomada); `registry_dir` separado; campo `territorio` no corpus; **modo `--dry-run-rota`**; `is_complex` calibrado (find_tables / densidade de valores). |
| `shared_utils.py` | `rota_por_nome()` / `tipo_ato_por_nome()` — classificador de rota pelo nome do PDF; `_RELATORIO_FISCAL_TOKENS`. |
| `engine_worker.py` | **Guarda anti-thrash** (limita OMP/MKL/OPENBLAS a 4 no caminho CPU) — evita o estouro de threads do PaddlePaddle no WSL. |
| `limpeza_textos.py` | **Dedup cross-file pós-limpeza** (reforço do P-09): descarta arquivos que ficam idênticos após a limpeza. (P-08/P-09 já estavam implementados.) |
| `pyproject.toml` | paddle movido para o extra `[paddle]` e **pinado `paddleocr>=2.10,<3`**; `opencv-python-headless` adicionado ao extra `docling`. |
| `setup_venvs.sh` (**novo**) | Cria os **dois venvs isolados**, com verificação, troca automática para `polars-lts-cpu` em CPUs sem AVX2, e instalação do paddle-gpu separada do paddleocr. |
| `MANUAL_EQUIPES.md` | Comandos atualizados (`./.venv/bin/python`, nunca `uv run`), tabela de roteamento nova, regra de datas, `--dry-run-rota`. |

### 1.2 Calibração do roteamento (medida em dados reais)
Medição em **1.028 documentos**: rotular *toda* licitação/contrato como "tem tabela" estava
errado (só **52%** das licitações e **16%** dos contratos têm tabela real; já **LRF/LOA = ~100%**).
Recalibrado para que o **motor leve seja maioria**:
- **Resultado:** ~**59% leve** (PaddleOCR/PyMuPDF) / ~**40% Docling**.

### 1.3 Deploy e validação no cluster
- Repositório + dataset (**20 GB / 13.293 PDFs**) enviados ao NFS do cluster.
- Os **dois venvs** construídos no servidor (torch-cu13 + Docling; paddle-gpu-cu126 + paddleocr 2.10).
- **Job SLURM real (gpunode01)**: Docling-CUDA **e** PaddleOCR-CUDA rodando, **0 erros**.

---

## 2. Como o pipeline funciona agora

```
PDFs em territorios/<slug>/pdfs/
        │
        ▼  extrair_territorio.py  (monta manifesto: SHA-256, município/entidade da pasta)
        │
        ▼  orquestrador_extracao.py
        │
   ┌────┴───────────── Triagem DLA (PyMuPDF, in-process) ─────────────┐
   │  • mapeia páginas por município   • score de texto nativo         │
   │  • detecta tabela no conteúdo (find_tables / valores monetários)  │
   │  • lê do NOME: rota (fiscal/comum) e ANO de publicação            │
   └──────────────────────────────────────────────────────────────────┘
        │
        ▼  Dedup PRÉ-extração (hash do texto de triagem) → pula duplicata sem pagar motor
        │
        ▼  ROTEAMENTO
        │     COM GPU:  comum → PaddleOCR-CUDA (worker .venv-paddle)
        │               fiscal/tabela → Docling-CUDA (worker .venv)
        │               escaneado → PaddleOCR-CUDA
        │     SEM GPU:  comum → PyMuPDF   fiscal → Docling-CPU   escaneado → Tesseract
        │
        ▼  Dedup PÓS-extração (hash do Markdown) → grava .md + corpus NDJSON
        │
        ▼  extraidos/<slug>/datalake/ano=AAAA/mes=sem_mes/municipio=.../<hash>.md
           extraidos/<slug>/corpus_<slug>.jsonl   (+ registros de dedup)
```

**Por que dois venvs?** `torch` (build cu13, Docling) e `paddlepaddle-gpu` (build cu126,
PaddleOCR) sobrescrevem as mesmas libs CUDA e **não coexistem** no mesmo ambiente. Cada motor
roda em **subprocesso isolado** (`engine_worker.py`), no seu venv, fixado numa GPU via
`CUDA_VISIBLE_DEVICES`. O orquestrador roda no `.venv` e dispara os workers.

**Regra de ouro:** **nunca** use `uv run` para extrair (re-sincroniza o `.venv` e pode
reinstalar paddle, quebrando o torch). Use os interpretadores diretos.

**Datas:** determinadas **só pelo nome do arquivo** (`extrair_data_filename`). Partição
`ano=AAAA/mes=sem_mes` (mês/dia exige o mapa edição→data, ainda pendente — ver §8).

---

## 3. Manual de uso

### 3.1 Preparação do ambiente (uma vez por máquina)
```bash
# Local (WSL) — paddle em CPU; ou no cluster sem --paddle-gpu
bash setup_venvs.sh

# Servidor de GPU (lab, cu126)
bash setup_venvs.sh --paddle-gpu

# opções: --force (recria do zero) | --verbose (set -x)
```
Cria `.venv` (torch + docling + orquestrador) e `.venv-paddle` (paddleocr isolado), com
verificação de imports ao final.

### 3.2 Comando padrão de extração
```bash
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio <slug> [opções]
```

| Parâmetro | Padrão | Função |
|---|---|---|
| `--territorio SLUG` | — | **Obrigatório.** Slug do território (§6). |
| `--modo` | `paddle` | `paddle` = orquestrador híbrido (padrão); `pymu` = PyMuPDF puro (rápido, sem OCR). |
| `--limite N` | ilimitado | Processa no máx. N PDFs (testes/lotes). |
| `--dry-run-rota` | off | **Só classifica a rota** de cada doc e gera `relatorio_rota.ndjson`; NÃO roda motores. |
| `--threshold` | `0.45` | Corte de texto nativo (abaixo → escaneado). |
| `--docling-max-paginas` | `8` | Cap anti-OOM: fatia o Docling em lotes de N páginas. |
| `--gpu-paddle` | `auto` | GPU do worker PaddleOCR: índice (`0`/`1`), `cpu`, ou `auto`. |
| `--gpu-docling` | `auto` | GPU do worker Docling. |
| `--verbose` | off | DEBUG no console (o `.log` já grava DEBUG sempre). |
| `--listar` | — | Lista os territórios e sai. |

**Exemplos:**
```bash
# Validar a separação de rota ANTES de gastar GPU (recomendado)
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio tabuleiros_alto_parnaiba --dry-run-rota

# Teste rápido (5 PDFs) com log detalhado
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio carnaubais --limite 5 --verbose

# WSL local (paddle só em CPU): mande o paddle para CPU
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio carnaubais --gpu-paddle cpu
```

### 3.3 Limpeza (pós-extração)
```bash
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.limpeza_textos \
    --input-dir extraidos/<slug>/datalake --output-dir dados_limpos/<slug> --debug
```
Aplica higienização, separa severidades de `needs_human_review` e **descarta duplicatas
pós-limpeza** (re-hash do conteúdo limpo).

### 3.4 Saídas geradas
```
extraidos/<slug>/
├── datalake/ano=AAAA/mes=sem_mes/municipio=<slug>/<hash>.md   ← Markdown + frontmatter
├── corpus_<slug>.jsonl        ← corpus para RAG/treino (schema unificado)
├── registro_dedup.ndjson      ← dedup pós-extração (persistente)
├── registro_dla_dedup.txt     ← dedup pré-extração (retomada)
├── relatorio_rota.ndjson      ← (se --dry-run-rota) classificação por doc
└── download_manifest.json
logs/<slug>/extracao_AAAAMMDD_HHMMSS.log   ← log DEBUG completo (auditoria)
```
**Schema do `corpus_<slug>.jsonl`:** `id_publicacao, territorio, municipio, tipo_ato,
data_publicacao, extrator, texto, n_chars`.

---

## 4. Verificação pelo terminal

### 4.1 Os venvs estão sãos?
```bash
# .venv (orquestrador + Docling)
./.venv/bin/python -c "import torch,polars,fitz,docling; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())"

# .venv-paddle (worker PaddleOCR) — em CPU pode falhar o import do paddle-gpu; valide no nó GPU
./.venv-paddle/bin/python -c "import paddle; from paddleocr import PPStructure; \
print('paddle',paddle.__version__,'cuda',paddle.is_compiled_with_cuda())"

# listar pacotes de um venv do uv (uv venv NÃO tem pip)
uv pip list --python .venv-paddle/bin/python | grep -iE 'paddle|opencv|pymupdf'
```

### 4.2 A GPU está visível?
```bash
./.venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
# nvidia-smi pode estar quebrado (NVML mismatch) e ainda assim o compute funcionar.
```

### 4.3 A extração produziu o esperado?
```bash
slug=tabuleiros_alto_parnaiba
# nº de .md gerados
find extraidos/$slug/datalake -name '*.md' | wc -l
# distribuição de extratores no corpus
./.venv/bin/python -c "
import polars as pl
df = pl.read_ndjson('extraidos/$slug/corpus_$slug.jsonl')
print('total', df.height)
print(df.group_by('extrator').len().sort('len', descending=True))
print('campos vazios:', {c: df.filter(pl.col(c).is_null()|(pl.col(c)=='')).height for c in ['municipio','tipo_ato','data_publicacao','texto']})
"
# rota planejada (sem rodar motores)
./.venv/bin/python -c "
import json,collections
c=collections.Counter(json.loads(l)['rota_final'] for l in open('extraidos/$slug/relatorio_rota.ndjson'))
print('rotas:', dict(c))"
```

---

## 5. Como funcionam os jobs (SLURM)

O cluster usa **SLURM**. O acesso à GPU **não é por `ssh` direto** ao nó — é via fila.

### 5.1 Conceitos
- **Partições:** `gpu` (padrão), `long`, `debug`.
- **Nós:** `gpunode01` (2× L4), `gpunode02` (1× L4). Cada **L4 = 24 GB VRAM**.
- **GRES:** recurso de GPU. Pede-se com `--gres=gpu:l4:N` (N = nº de GPUs).
- **`sbatch`** (fila/lote, recomendado) vs **`srun`** (interativo, fica preso esperando alocação).

### 5.2 Anatomia de um job (`job_extracao.sbatch`)
```bash
#!/bin/bash
#SBATCH --job-name=dompi_<slug>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:2        # 2 GPUs → paddle na GPU0, docling na GPU1 (paralelo)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G              # ⚠️ OBRIGATÓRIO (ver armadilha abaixo)
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log
set -e
cd ~/building-a-LLM
PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio \
    --territorio <slug>          # sem --limite = território inteiro
```
```bash
sbatch job_extracao.sbatch
```

> ⚠️ **Armadilha do `--mem`:** sem `--mem`, o SLURM reserva **a RAM inteira do nó (~62 GB)** e o
> job fica `PENDING (Resources)` se houver qualquer outro job. **Sempre defina `--mem=32G`**
> (Docling tem pico ~12 GB; 32 GB dá folga e convive com outros jobs).

### 5.3 Acompanhamento
```bash
squeue -u $USER -o "%.6i %.9P %.12j %.2t %.10M %R"   # PD=pendente, R=rodando
scontrol show job <ID> | tr ' ' '\n' | grep -iE 'JobState|Reason|TRES'   # por que está PD
scancel <ID>                                          # cancelar
tail -f <job-name>_<ID>.log                           # acompanhar a saída
```

### 5.4 Roteiro de submissão validado
```bash
ssh aluno_matheus@10.94.80.10
cd ~/building-a-LLM
sbatch job_extracao.sbatch && squeue -u $USER
# quando ST=R, acompanhe: tail -f dompi_<slug>_<ID>.log
```

---

## 6. Checklist de execução por território

São **13 territórios de desenvolvimento**. Hoje só o `tabuleiros_alto_parnaiba` tem PDFs
no servidor; os demais entram conforme as equipes depositarem os arquivos.

| # | Slug (no comando) | Nome canônico | Status PDFs |
|---|---|---|---|
| 1 | `planice_litoran` | Planície Litorânea | ☐ |
| 2 | `cocais` | Cocais | ☐ |
| 3 | `carnaubais` | Carnaubais | ☐ |
| 4 | `entre_rios` | Entre Rios | ☐ |
| 5 | `vale_do_sambito` | Vale do Sambito | ☐ |
| 6 | `vale_do_rio_guaribas` | Vale do Rio Guaribas | ☐ |
| 7 | `chapada_vale_do_rio_itaim` | Chapada Vale do Rio Itaim | ☐ |
| 8 | `vale_do_caninde` | Vale do Canindé | ☐ |
| 9 | `serra_da_capivara` | Serra da Capivara | ☐ |
| 10 | `vale_dos_rios_piaui_e_itaueiras` | Vale dos Rios Piauí e Itaueiras | ☐ |
| 11 | `tabuleiros_alto_parnaiba` | Tabuleiros do Alto Parnaíba e Chapada das Mangabeiras | ✅ 13.293 PDFs |
| 12 | `teresina` | Teresina | ☐ |
| 13 | `parnaiba` | Parnaíba | ☐ |

### Checklist por território (repetir para cada um)
```
[ ] 1. PDFs depositados em territorios/<slug>/pdfs/ (subpastas município/entidade ajudam)
[ ] 2. Enviar ao cluster (rsync paralelo por subpasta — caminho ABSOLUTO, --partial):
       cd territorios/<slug>/pdfs
       ls -d */ | sed 's#/##' | xargs -P 8 -I{} \
         rsync -a --partial "{}/" "aluno_matheus@10.94.80.10:/home/aluno_matheus/building-a-LLM/territorios/<slug>/pdfs/{}/"
[ ] 3. Conferir integridade: contagem local == contagem remota (find ... -name '*.pdf' | wc -l)
[ ] 4. DRY-RUN de rota (rápido, sem GPU): --dry-run-rota → revisar relatorio_rota.ndjson
[ ] 5. Teste pequeno: sbatch com --limite 10 (validar 0 erros, extratores corretos)
[ ] 6. Run completo: sbatch sem --limite, --mem=32G --gres=gpu:l4:2
[ ] 7. Verificar saída (§4.3): nº de .md, distribuição de extratores, campos vazios
[ ] 8. Limpeza: limpeza_textos.py → dados_limpos/<slug>
[ ] 9. (Opcional) Reexecutar o mesmo job para confirmar idempotência (deve pular ~100%)
```

> **Idempotência:** re-rodar o mesmo território pula o que já foi extraído (via
> `registro_dla_dedup.txt`). Seguro para retomar após interrupção.

---

## 7. Recursos disponíveis

**Hardware (cluster SLURM `ncad.ufpi.br`)**
- `gpunode01`: **2× NVIDIA L4 (24 GB cada)**, 16 cores, 62 GB RAM.
- `gpunode02`: 1× L4, 12 cores.
- `/home` em **NFS** (1,8 TB livres), **compartilhado** entre master e nós → transferir 1×.
- `slurm-master`: nó de login (CPU, sem AVX2 — por isso `polars-lts-cpu`).

**Software / motores de extração**
- **PyMuPDF** — texto nativo (instantâneo, in-process).
- **Docling-CUDA** — tabelas/relatórios fiscais em Markdown estruturado.
- **PaddleOCR-CUDA** (PP-Structure 2.10) — documentos comuns e escaneados.
- **Tesseract** — escaneado em CPU (fallback sem GPU).

**Ferramentas operacionais**
- `setup_venvs.sh` — monta os dois ambientes (idempotente).
- `--dry-run-rota` — valida a separação fiscal/comum sem custo de GPU.
- **Dedup em camadas** — PDF/URL (scraping) → pré-extração → pós-extração → pós-limpeza.
- **Retomada idempotente** — registros persistentes.
- `monitor_os.sh` — monitor de RAM/CPU para execuções longas.

---

## 8. Possíveis aprimoramentos

**Qualidade dos dados**
- **Mapa edição→data (P-03):** capturar a data da edição no scraping para preencher
  `mes`/`dia` (hoje `mes=sem_mes`); habilita filtros temporais no RAG.
- **Reconhecimento do município** em PDFs sem cabeçalho legível (hoje cai em fallback).

**Desempenho / escala**
- **Usar as 2 GPUs do gpunode01 em paralelo** no run completo (`--gres=gpu:l4:2`): paddle na
  GPU0, docling na GPU1 — dobra o throughput dos lotes mistos.
- **Job array do SLURM** (um job por município ou por faixa) para paralelizar entre nós.
- **Pré-cache dos modelos** (Docling/PaddleOCR) num job inicial, já que o `~/.cache` é NFS
  compartilhado — elimina o custo de download na 1ª execução de cada nó.
- Avaliar **Docling com `do_ocr=True`** só nas páginas realmente escaneadas (raras).

**Robustez operacional**
- Tornar a **verificação do paddle-gpu não-fatal em nós CPU** no `setup_venvs.sh` (hoje ela
  "falha" no master por falta de driver, embora os pacotes instalem corretamente).
- **Checkpoint/limite de tempo**: jobs longos com `--time` adequado e retomada automática.
- **Tabela de QOS/limites** do cluster documentada (evita `PENDING` por cota).

**Governança do repositório**
- **Commit das mudanças** (sem assinatura) na `feat-paddle-unificacao` e abertura de PR.
- **Limpeza dos 8 commits antigos** com assinatura Claude (requer `force-push`; combinar com o time).

---

*Documento gerado na sessão de 2026-06-01. Para detalhes de schema e troubleshooting de
extração ver também `MANUAL_EQUIPES.md`; para o benchmark dos motores ver `docs/BENCHMARK_OCR.md`.*
