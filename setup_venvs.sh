#!/usr/bin/env bash
# =============================================================================
# setup_venvs.sh — Monta os DOIS ambientes isolados da extração DOM-PI
# -----------------------------------------------------------------------------
# Por que dois venvs? `torch` (build cu13, usado por Docling) e
# `paddlepaddle-gpu` (build cu126, usado por PaddleOCR) instalam libs no mesmo
# namespace `nvidia/` (libnccl/libcudnn) e se sobrescrevem — instalar paddle-gpu
# no mesmo venv QUEBRA o `import torch` (undefined symbol: ncclCommWindowDeregister).
# Por isso cada engine roda em SEU venv, em SEU subprocesso (engine_worker.py):
#
#   .venv         → torch + docling + pymupdf + polars  (orquestrador + Docling)
#   .venv-paddle  → paddlepaddle(-gpu) + paddleocr + pymupdf  (worker PaddleOCR)
#
# REGRA DE OURO: nunca use `uv run` para a extração — ele re-sincroniza o `.venv`
# e pode reinstalar paddle, requebrando o torch. Invoque os interpretadores
# diretamente: ./.venv/bin/python e ./.venv-paddle/bin/python.
#
# Uso:
#   bash setup_venvs.sh                 # paddle CPU (universal; recomendado p/ WSL)
#   bash setup_venvs.sh --paddle-gpu    # paddle GPU build (lab com 2x L4 / cu126)
#   bash setup_venvs.sh --force         # recria os venvs do zero
#   bash setup_venvs.sh --verbose       # log de cada comando (set -x)
#
# Tudo é idempotente: rodar de novo só completa o que falta (a menos de --force).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- args / flags
PADDLE_GPU=0
FORCE=0
VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --paddle-gpu) PADDLE_GPU=1 ;;
        --force)      FORCE=1 ;;
        --verbose)    VERBOSE=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -40
            exit 0 ;;
        *) echo "[setup_venvs] Argumento desconhecido: $arg" >&2; exit 2 ;;
    esac
done
[ "$VERBOSE" -eq 1 ] && set -x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ----------------------------------------------------------------- logging util
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '%s [setup_venvs] %s\n' "$(ts)" "$*"; }
hr()  { printf '%s\n' "============================================================"; }

hr
log "Início. ROOT=$ROOT"
log "Opções: paddle_gpu=$PADDLE_GPU force=$FORCE verbose=$VERBOSE"
hr

# ------------------------------------------------------------------ pré-checagem
if ! command -v uv >/dev/null 2>&1; then
    log "ERRO: 'uv' não encontrado no PATH. Instale: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
log "uv: $(command -v uv) ($(uv --version 2>&1))"
log "python (host): $(python3 --version 2>&1 || echo 'n/d')"

# Diagnóstico de hardware (apenas informativo; não falha o setup)
if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi detectado:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | sed 's/^/    GPU: /' || log "    (nvidia-smi falhou — NVML pode estar quebrado; compute CUDA ainda pode funcionar)"
else
    log "nvidia-smi ausente (normal em WSL sem driver exposto; torch ainda pode ver a GPU)"
fi
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    log "Ambiente: WSL detectado."
fi

# ============================================================================
# 1) .venv  — torch + docling + projeto (engine Docling + orquestrador)
# ============================================================================
hr
log "[1/2] Montando .venv (torch + docling + projeto)"
if [ "$FORCE" -eq 1 ] && [ -d .venv ]; then
    log "    --force: removendo .venv existente"
    rm -rf .venv
fi
# `uv sync` cria/atualiza o .venv a partir do pyproject; --extra docling puxa o Docling.
log "    Executando: uv sync --extra docling"
uv sync --extra docling 2>&1 | sed 's/^/    [uv] /'

# COMPATIBILIDADE DE CPU: o wheel padrão do Polars exige AVX/AVX2. Nós de cluster
# antigos (ex.: slurm-master) não têm → Polars crasha com SIGILL. Se o CPU de
# BUILD não tiver AVX2, troca por `polars-lts-cpu` (roda em qualquer x86_64,
# inclusive nos nós de GPU). Idempotente: refaz a troca a cada `uv sync`.
if ! grep -qm1 '\bavx2\b' /proc/cpuinfo 2>/dev/null; then
    log "    CPU sem AVX2 → substituindo polars por polars-lts-cpu (compatibilidade de nó)"
    uv pip uninstall --python .venv/bin/python polars 2>&1 | sed 's/^/    [uv pip] /' || true
    uv pip install   --python .venv/bin/python "polars-lts-cpu>=1.0.0" 2>&1 | sed 's/^/    [uv pip] /'
else
    log "    CPU com AVX2 → polars padrão OK"
fi

log "    Verificando imports críticos no .venv ..."
if ./.venv/bin/python - <<'PY' 2>&1 | sed 's/^/    [.venv] /'
import sys
def chk(mod, extra=""):
    try:
        m = __import__(mod)
        print(f"OK   {mod} {getattr(m,'__version__','?')} {extra}")
        return True
    except Exception as e:
        print(f"FALHA {mod}: {e}")
        return False
ok = True
ok &= chk("torch")
try:
    import torch
    print(f"INFO torch.cuda.is_available()={torch.cuda.is_available()} device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"INFO GPU[0]={p.name} VRAM={p.total_memory/1e9:.1f}GB")
except Exception as e:
    print(f"INFO torch.cuda inspeção falhou: {e}")
ok &= chk("fitz", "(pymupdf)")
ok &= chk("polars")
try:
    import docling; print("OK   docling")
except Exception as e:
    print(f"AVISO docling indisponível ({e}) — rota de tabelas cairá em fallback")
sys.exit(0 if ok else 1)
PY
then
    log "    .venv OK (torch + pymupdf + polars verificados)"
else
    log "    ERRO: verificação do .venv falhou (ver linhas acima)."
    exit 1
fi

# ============================================================================
# 2) .venv-paddle  — paddlepaddle(-gpu) + paddleocr (worker PaddleOCR isolado)
# ============================================================================
hr
log "[2/2] Montando .venv-paddle (paddleocr isolado)"
if [ "$FORCE" -eq 1 ] && [ -d .venv-paddle ]; then
    log "    --force: removendo .venv-paddle existente"
    rm -rf .venv-paddle
fi
if [ ! -d .venv-paddle ]; then
    log "    Criando venv: uv venv .venv-paddle --python 3.12"
    uv venv .venv-paddle --python 3.12 2>&1 | sed 's/^/    [uv] /'
else
    log "    .venv-paddle já existe (use --force para recriar)"
fi

PADDLE_PKG="paddlepaddle>=3.3.1"
if [ "$PADDLE_GPU" -eq 1 ]; then
    # Build GPU (cu126). Requer o índice do PaddlePaddle p/ a CUDA correta.
    PADDLE_PKG="paddlepaddle-gpu>=3.3.1"
    log "    Modo GPU: instalando $PADDLE_PKG (cu126)."
    log "    NOTA: se o wheel cu126 não casar com sua CUDA, ajuste o índice em https://www.paddlepaddle.org.cn/install"
    # Dois comandos: paddle-gpu vem do índice cu126; paddleocr<3 + pymupdf vêm do
    # PyPI. Num comando só, o uv barra por "index confusion" (o índice cu126 tem
    # paddleocr apenas em 3.x, e o uv não busca o <3 no PyPI por padrão).
    uv pip install --python .venv-paddle/bin/python "$PADDLE_PKG" \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ 2>&1 | sed 's/^/    [uv pip] /'
    uv pip install --python .venv-paddle/bin/python "paddleocr>=2.10,<3" "pymupdf>=1.27.2.2" 2>&1 | sed 's/^/    [uv pip] /'
else
    log "    Modo CPU (universal): instalando $PADDLE_PKG + paddleocr + pymupdf."
    log "    Em WSL/1-GPU isto basta: escaneados são raros e o orquestrador roda paddle com --gpu-paddle cpu."
    uv pip install --python .venv-paddle/bin/python "$PADDLE_PKG" "paddleocr>=2.10,<3" "pymupdf>=1.27.2.2" 2>&1 | sed 's/^/    [uv pip] /'
fi

log "    Verificando imports no .venv-paddle ..."
if ./.venv-paddle/bin/python - <<'PY' 2>&1 | sed 's/^/    [.venv-paddle] /'
import sys
ok = True
try:
    import paddle
    print(f"OK   paddle {paddle.__version__}")
    print(f"INFO paddle.is_compiled_with_cuda()={paddle.is_compiled_with_cuda()}")
except Exception as e:
    print(f"FALHA paddle: {e}"); ok = False
try:
    import paddleocr
    print(f"OK   paddleocr {getattr(paddleocr,'__version__','?')}")
except Exception as e:
    print(f"FALHA paddleocr: {e}"); ok = False
try:
    import fitz
    print(f"OK   fitz (pymupdf) {fitz.__doc__.splitlines()[0] if fitz.__doc__ else ''}")
except Exception as e:
    print(f"FALHA fitz: {e}"); ok = False
# Garantia anti-conflito: o torch NÃO deve estar neste venv (senão volta o choque de libs)
try:
    import torch  # noqa: F401
    print("AVISO torch presente no .venv-paddle — risco de conflito de libs nvidia/. Mantê-los separados.")
except Exception:
    print("OK   torch ausente do .venv-paddle (correto — engines isoladas)")
sys.exit(0 if ok else 1)
PY
then
    log "    .venv-paddle OK"
else
    log "    ERRO: verificação do .venv-paddle falhou (ver linhas acima)."
    exit 1
fi

# ============================================================================
hr
log "CONCLUÍDO. Ambientes prontos:"
log "    .venv         → $(./.venv/bin/python --version 2>&1)   (torch + docling + orquestrador)"
log "    .venv-paddle  → $(./.venv-paddle/bin/python --version 2>&1)   (paddleocr isolado)"
hr
log "Próximo passo (NUNCA use 'uv run' para extrair):"
log "    PYTHONPATH=src ./.venv/bin/python -m dompi_scraper.extrair_territorio --territorio carnaubais --limite 3 --verbose"
hr
