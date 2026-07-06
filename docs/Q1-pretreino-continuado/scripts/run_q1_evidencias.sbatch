#!/bin/bash
#SBATCH --job-name=q1-evidencias
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=trabalho-final/Q1-pretreino-continuado/resultados/log_evidencias_%j.txt
#SBATCH --error=trabalho-final/Q1-pretreino-continuado/resultados/log_evidencias_%j.txt

# Q1 — Evidências do resultado de DAPT (camadas de avaliação de continued-pretraining).
# Só INFERÊNCIA (sem treino). Submeter da RAIZ do repo: ~/building-a-LLM
#
# Produz três frentes para o relatório:
#   1. RETENÇÃO DE CAPACIDADE — PPL/CE num held-out de DOMÍNIO GERAL (Wikipedia-PT).
#   2. GANHO DE DOMÍNIO — PPL/CE no held-out de DOMÍNIO (DOM-PI) + acurácia de GERAÇÃO
#      no benchmark (exact-match / contains / token-F1), com a resposta COMPLETA salva.
#   3. CURVA DOSE-RESPOSTA — figura PPL x tokens.
#
# Modelos (baseline vs. os dois DAPT publicados no HF):
#   baseline  = Qwen/Qwen2.5-1.5B
#   unificado = gutoportelaa/qwen2.5-1.5b-dompi-fullft-unificado
#   teresina  = gutoportelaa/qwen2.5-1.5b-dompi-teresina-v3

set -e
REPO="$HOME/building-a-LLM"
cd "$REPO"
PYTHON="$REPO/.venv/bin/python3"
Q1="$REPO/trabalho-final/Q1-pretreino-continuado"
SCR="$Q1/scripts"
BENCH="$Q1/dompi_qa.jsonl"
RES="$Q1/resultados"
HELDOUT_GERAL="$Q1/data/held_out_geral.jsonl"
HELDOUT_UNIF="$REPO/data/held_out.jsonl"
HELDOUT_TER="$REPO/data/teresina/held_out.jsonl"
mkdir -p "$RES/figuras" "$Q1/data"

# modelo -> held-out de domínio correspondente
declare -A MODELS=(
  [baseline]="Qwen/Qwen2.5-1.5B"
  [unificado]="gutoportelaa/qwen2.5-1.5b-dompi-fullft-unificado"
  [teresina]="gutoportelaa/qwen2.5-1.5b-dompi-teresina-v3"
)
declare -A DOMAIN_HELDOUT=(
  [baseline]="$HELDOUT_UNIF"
  [unificado]="$HELDOUT_UNIF"
  [teresina]="$HELDOUT_TER"
)

echo "=== Q1 — Evidências de DAPT ==="
echo "Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[aviso] nvidia-smi indisponível"
$PYTHON -c "import datasets, matplotlib" 2>/dev/null || $PYTHON -m pip install datasets matplotlib --quiet

# 0. held-out de domínio geral (uma vez)
if [ ! -f "$HELDOUT_GERAL" ]; then
  echo "--- Construindo held-out de domínio geral (Wikipedia-PT)..."
  $PYTHON "$SCR/preparar_heldout_geral.py" --n 1000 --output "$HELDOUT_GERAL"
fi

# 1+2. retenção (geral) + domínio (DOM-PI) + acurácia de geração, por modelo
for label in baseline unificado teresina; do
  model="${MODELS[$label]}"
  echo "=================================================================="
  echo ">>> Modelo: $label ($model)"

  echo "--- [retenção] held-out de domínio GERAL"
  $PYTHON "$SCR/avaliar_modelo.py" --model "$model" \
      --held-out "$HELDOUT_GERAL" --benchmark "$BENCH" \
      --output "$RES/retencao_${label}.json" --max-samples 1000

  echo "--- [ganho de domínio] held-out DOM-PI"
  $PYTHON "$SCR/avaliar_modelo.py" --model "$model" \
      --held-out "${DOMAIN_HELDOUT[$label]}" --benchmark "$BENCH" \
      --output "$RES/dominio_${label}.json" --max-samples 1000

  echo "--- [ganho de domínio] acurácia de GERAÇÃO no benchmark (resposta completa salva)"
  $PYTHON "$SCR/avaliar_geracao.py" --model "$model" \
      --benchmark "$BENCH" --output "$RES/geracao_${label}.json"
done

# 3. curva dose-resposta
echo "--- [dose-resposta] gerando figura PPL x tokens"
$PYTHON "$SCR/curva_dose_resposta.py" \
    --points "$RES/curva_pontos.json" --output "$RES/figuras/curva_dose_resposta.png"

echo "=================================================================="
echo "RESUMO RETENÇÃO (PPL held-out GERAL — quanto menor a alta vs baseline, melhor):"
for label in baseline unificado teresina; do
  $PYTHON -c "import json;d=json.load(open('$RES/retencao_${label}.json'));print(f'  ${label}: PPL_geral={d[\"perplexity\"]} CE={d[\"cross_entropy\"]} tokacc={d[\"token_accuracy\"]}')"
done
echo "RESUMO DOMÍNIO (PPL held-out DOM-PI):"
for label in baseline unificado teresina; do
  $PYTHON -c "import json;d=json.load(open('$RES/dominio_${label}.json'));print(f'  ${label}: PPL_dominio={d[\"perplexity\"]} CE={d[\"cross_entropy\"]} tokacc={d[\"token_accuracy\"]}')"
done
echo "RESUMO GERAÇÃO (contains / token_f1 médio / exact-match):"
for label in baseline unificado teresina; do
  $PYTHON -c "import json;d=json.load(open('$RES/geracao_${label}.json'));print(f'  ${label}: contains={d[\"gen_contains\"]} f1_medio={d[\"gen_f1_mean\"]} EM={d[\"gen_exact_match\"]}')"
done
echo "=== Concluído. JSONs em $RES/ | figura em $RES/figuras/ ==="
