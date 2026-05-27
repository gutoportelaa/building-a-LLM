#!/usr/bin/env bash
# =============================================================================
# setup_territorios.sh — Cria a estrutura de diretórios padronizada DOM-PI
# =============================================================================
# Uso: bash setup_territorios.sh
# Execute a partir da raiz do projeto (building-a-LLM/).
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Slugs padronizados dos 13 territórios
TERRITORIOS=(
    "planice_litoran"
    "cocais"
    "carnaubais"
    "entre_rios"
    "vale_do_sambito"
    "vale_do_rio_guaribas"
    "chapada_vale_do_rio_itaim"
    "vale_do_caninde"
    "serra_da_capivara"
    "vale_dos_rios_piaui_e_itaueiras"
    "tabuleiros_alto_parnaiba"
    "teresina"
    "parnaiba"
)

echo "============================================================"
echo " DOM-PI — Criando estrutura de territórios"
echo " Raiz: ${ROOT}"
echo "============================================================"

for slug in "${TERRITORIOS[@]}"; do
    # Drop-zone: onde a equipe deposita os PDFs coletados
    mkdir -p "${ROOT}/territorios/${slug}/pdfs"
    # Saída da extração (gerada pelo script de extração)
    mkdir -p "${ROOT}/extraidos/${slug}"
    # Logs por território
    mkdir -p "${ROOT}/logs/${slug}"
    echo "  ✅ territorios/${slug}/"
done

# Diretório de corpus final consolidado
mkdir -p "${ROOT}/corpus_final"

echo ""
echo "Estrutura criada com sucesso:"
echo ""
echo "  territorios/                ← DROP-ZONE: equipes depositam PDFs aqui"
for slug in "${TERRITORIOS[@]}"; do
    echo "    ${slug}/pdfs/"
done
echo ""
echo "  extraidos/                  ← Gerado automaticamente pelo extrator"
echo "  logs/                       ← Logs por território"
echo "  corpus_final/               ← corpus_unificado.jsonl e relatórios"
echo ""
echo "Próximo passo: veja MANUAL_EQUIPES.md para instruções de extração."
