#!/bin/bash
##############################################################################
# pipeline_otimizado.sh — Pipeline DOM-PI Completo Otimizado
##############################################################################
# Orquestra as 3 etapas com paralelização:
# 1. Scraping paralelo (asyncio + aiohttp)
# 2. Download incrementalizado de PDFs
# 3. Extração com Marker (GPU-accelerated)
#
# Uso:
#   bash pipeline_otimizado.sh [options]
#
# Opções:
#   --territorio-carnaubais    Todos os 16 municípios (padrão)
#   --municipio "Campo Maior"  Município específico
#   --ano 2025                Ano (padrão: 2025)
#   --max-concorrencia 15     Máx requisições paralelas (padrão: 15)
#   --limite-pdfs 999999      Máx PDFs a baixar (padrão: ilimitado)
#   --skip-scraping           Pula etapa 1 (usa JSON existente)
#   --skip-download           Pula etapa 2 (usa PDFs existentes)
#   --skip-extraction         Pula etapa 3
#   --dry-run                 Simula sem efetivamente fazer
#   --verbose                 Ativa logs DEBUG
#
# Exemplo completo:
#   bash pipeline_otimizado.sh --territorio-carnaubais --ano 2025 --max-concorrencia 20 --verbose
#
##############################################################################

set -e  # Exit on error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Configurações padrão
TERRITORIO_CARNAUBAIS=true
MUNICIPIO=""
ANO=2025
MAX_CONCORRENCIA=15
LIMITE_PDFS=999999
SKIP_SCRAPING=false
SKIP_DOWNLOAD=false
SKIP_EXTRACTION=false
DRY_RUN=false
VERBOSE=false

# Parâmetros do ambiente
WORKSPACE_ROOT="/home/gutemberg/Documents/building-a-LLM"
BASE_OUTPUT="${WORKSPACE_ROOT}/db_treino_carnaubais"
SCRAPING_BASE="scraping_carnaubais_${ANO}"
LOG_DIR="${BASE_OUTPUT}/logs"

# Função de uso
usage() {
    echo "Uso: bash pipeline_otimizado.sh [options]"
    echo ""
    echo "Opções:"
    echo "  --territorio-carnaubais      Todos os 16 municípios (padrão)"
    echo "  --municipio 'Nome'           Município específico"
    echo "  --ano YYYY                   Ano (padrão: 2025)"
    echo "  --max-concorrencia N         Máx requisições paralelas (padrão: 15)"
    echo "  --limite-pdfs N              Máx PDFs a baixar (padrão: ilimitado)"
    echo "  --skip-scraping              Pula etapa 1 (usa JSON existente)"
    echo "  --skip-download              Pula etapa 2 (usa PDFs existentes)"
    echo "  --skip-extraction            Pula etapa 3"
    echo "  --dry-run                    Simula sem executar"
    echo "  --verbose                    Ativa logs DEBUG"
    echo ""
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --territorio-carnaubais)
            TERRITORIO_CARNAUBAIS=true
            MUNICIPIO=""
            shift
            ;;
        --municipio)
            MUNICIPIO="$2"
            TERRITORIO_CARNAUBAIS=false
            shift 2
            ;;
        --ano)
            ANO="$2"
            SCRAPING_BASE="scraping_carnaubais_${ANO}"
            shift 2
            ;;
        --max-concorrencia)
            MAX_CONCORRENCIA="$2"
            shift 2
            ;;
        --limite-pdfs)
            LIMITE_PDFS="$2"
            shift 2
            ;;
        --skip-scraping)
            SKIP_SCRAPING=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Opção desconhecida: $1"
            usage
            ;;
    esac
done

# Função de log
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG_FILE}"
}

# Inicialização
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

log_info "========================================================================="
log_info "  DOM-PI Pipeline Otimizado — $(date '+%Y-%m-%d %H:%M:%S')"
log_info "========================================================================="
log_info "Workspace: ${WORKSPACE_ROOT}"
log_info "Output dir: ${BASE_OUTPUT}"
log_info "Log file: ${LOG_FILE}"

if [ "${DRY_RUN}" = true ]; then
    log_warn "⚠️  MODO DRY-RUN — Nenhuma execução real será realizada"
fi

log_info ""
log_info "Configurações:"
if [ "${TERRITORIO_CARNAUBAIS}" = true ]; then
    log_info "  - Alvo: Território Carnaubais (16 municípios)"
else
    log_info "  - Alvo: Município único: ${MUNICIPIO}"
fi
log_info "  - Ano: ${ANO}"
log_info "  - Max concorrência: ${MAX_CONCORRENCIA}"
log_info "  - Limite de PDFs: ${LIMITE_PDFS}"
log_info "  - Verbose: ${VERBOSE}"
log_info ""

# ============================================================================
# ETAPA 1: SCRAPING PARALELO
# ============================================================================

if [ "${SKIP_SCRAPING}" = true ]; then
    log_warn "Pulando etapa 1 (scraping) — usando JSON existente"
else
    log_info "========================================================================="
    log_info "ETAPA 1: SCRAPING PARALELO"
    log_info "========================================================================="
    
    cd "${WORKSPACE_ROOT}"
    
    # Constrói comando do scraper
    SCRAPER_CMD="uv run python src/dompi_scraper/scraper_otimizado_threads.py"
    
    if [ "${TERRITORIO_CARNAUBAIS}" = true ]; then
        SCRAPER_CMD="${SCRAPER_CMD} --territorio-carnaubais"
    else
        SCRAPER_CMD="${SCRAPER_CMD} --municipio '${MUNICIPIO}'"
    fi
    
    SCRAPER_CMD="${SCRAPER_CMD} --ano ${ANO}"
    SCRAPER_CMD="${SCRAPER_CMD} --max-workers ${MAX_CONCORRENCIA}"
    SCRAPER_CMD="${SCRAPER_CMD} --saida ${SCRAPING_BASE}"
    
    if [ "${VERBOSE}" = true ]; then
        SCRAPER_CMD="${SCRAPER_CMD} --verbose"
    fi
    
    log_info "Executando: ${SCRAPER_CMD}"
    
    if [ "${DRY_RUN}" = true ]; then
        log_warn "DRY-RUN: não executando scraper"
    else
        # Executa comando e captura saída + exit code
        set +e
        eval "${SCRAPER_CMD}" 2>&1 | tee -a "${LOG_FILE}"
        SCRAPER_EXIT=$?
        set -e
        
        if [ ${SCRAPER_EXIT} -ne 0 ]; then
            log_error "❌ Scraping falhou com exit code ${SCRAPER_EXIT}"
            exit 1
        fi
    fi
    
    log_success "✅ Etapa 1 concluída: scraping paralelo"
fi

# ============================================================================
# ETAPA 2: DOWNLOAD DE PDFs
# ============================================================================

if [ "${SKIP_DOWNLOAD}" = true ]; then
    log_warn "Pulando etapa 2 (download) — usando PDFs existentes"
else
    log_info ""
    log_info "========================================================================="
    log_info "ETAPA 2: DOWNLOAD INCREMENTAL DE PDFs"
    log_info "========================================================================="
    
    cd "${WORKSPACE_ROOT}"
    
    # Localiza JSON deduplicado
    JSON_INPUT="${SCRAPING_BASE}_deduplicados.json"
    
    if [ ! -f "${JSON_INPUT}" ]; then
        log_error "❌ Arquivo não encontrado: ${JSON_INPUT}"
        log_error "   Verifique se a etapa 1 foi executada com sucesso"
        exit 1
    fi
    
    # Constrói comando do downloader
    DOWNLOAD_CMD="uv run python src/dompi_scraper/download_pdfs.py"
    DOWNLOAD_CMD="${DOWNLOAD_CMD} --input ${JSON_INPUT}"
    DOWNLOAD_CMD="${DOWNLOAD_CMD} --output-dir ${BASE_OUTPUT}/pdfs_arquivos"
    DOWNLOAD_CMD="${DOWNLOAD_CMD} --manifest ${BASE_OUTPUT}/pdfs_arquivos/download_manifest.json"
    DOWNLOAD_CMD="${DOWNLOAD_CMD} --limite ${LIMITE_PDFS}"
    
    if [ "${VERBOSE}" = true ]; then
        DOWNLOAD_CMD="${DOWNLOAD_CMD} --verbose"
    fi
    
    log_info "Executando: ${DOWNLOAD_CMD}"
    
    if [ "${DRY_RUN}" = true ]; then
        log_warn "DRY-RUN: não executando download"
    else
        # Executa comando e captura saída + exit code
        set +e
        eval "${DOWNLOAD_CMD}" 2>&1 | tee -a "${LOG_FILE}"
        DOWNLOAD_EXIT=$?
        set -e
        
        if [ ${DOWNLOAD_EXIT} -ne 0 ]; then
            log_error "❌ Download falhou com exit code ${DOWNLOAD_EXIT}"
            exit 1
        fi
    fi
    
    log_success "✅ Etapa 2 concluída: download de PDFs"
fi

# ============================================================================
# ETAPA 3: EXTRAÇÃO COM MARKER
# ============================================================================

if [ "${SKIP_EXTRACTION}" = true ]; then
    log_warn "Pulando etapa 3 (extração) — conforme solicitado"
else
    log_info ""
    log_info "========================================================================="
    log_info "ETAPA 3: EXTRAÇÃO COM MARKER (GPU-Accelerated)"
    log_info "========================================================================="
    
    cd "${WORKSPACE_ROOT}"
    
    # Constrói comando do extrator
    EXTRATOR_CMD="uv run python src/dompi_scraper/extrator_marker.py"
    EXTRATOR_CMD="${EXTRATOR_CMD} --pasta-pdfs ${BASE_OUTPUT}/pdfs_arquivos"
    EXTRATOR_CMD="${EXTRATOR_CMD} --tamanho-amostra 999999"
    EXTRATOR_CMD="${EXTRATOR_CMD} --jsonl-output corpus_marker_${ANO}.jsonl"
    
    if [ "${VERBOSE}" = true ]; then
        EXTRATOR_CMD="${EXTRATOR_CMD} --verbose"
    fi
    
    log_info "Executando: ${EXTRATOR_CMD}"
    log_warn "⚠️  Esta etapa pode levar várias horas (GPU-bound)"
    
    if [ "${DRY_RUN}" = true ]; then
        log_warn "DRY-RUN: não executando extrator"
    else
        # Executa comando e captura saída + exit code
        set +e
        eval "${EXTRATOR_CMD}" 2>&1 | tee -a "${LOG_FILE}"
        EXTRATOR_EXIT=$?
        set -e
        
        if [ ${EXTRATOR_EXIT} -ne 0 ]; then
            log_error "❌ Extração falhou com exit code ${EXTRATOR_EXIT}"
            exit 1
        fi
    fi
    
    log_success "✅ Etapa 3 concluída: extração com Marker"
fi

# ============================================================================
# RESUMO FINAL
# ============================================================================

log_info ""
log_info "========================================================================="
log_info "✅ PIPELINE CONCLUÍDO COM SUCESSO"
log_info "========================================================================="
log_info "Tempo final: $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Log completo: ${LOG_FILE}"
log_info ""
log_info "Arquivos gerados:"
log_info "  - JSON bruto:    ${SCRAPING_BASE}.json"
log_info "  - JSON dedup:    ${SCRAPING_BASE}_deduplicados.json"
log_info "  - PDFs:          ${BASE_OUTPUT}/pdfs_arquivos/"
log_info "  - Manifesto:     ${BASE_OUTPUT}/pdfs_arquivos/download_manifest.json"
log_info "  - Corpus JSONL:  corpus_marker_${ANO}.jsonl"
log_info "========================================================================="
