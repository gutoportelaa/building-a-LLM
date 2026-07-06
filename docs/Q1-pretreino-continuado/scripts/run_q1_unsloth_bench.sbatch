#!/bin/bash
#SBATCH --job-name=q1-unsloth-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=trabalho-final/Q1-pretreino-continuado/resultados/log_unsloth_bench_%j.txt
#SBATCH --error=trabalho-final/Q1-pretreino-continuado/resultados/log_unsloth_bench_%j.txt

# Q1 — Benchmark Unsloth vs backend padrão (A/B controlado).
# Treina full-FT com Unsloth no MESMO subconjunto de 10% do corpus limpo que o
# job de dose-resposta padrão usou (data/train_limpo_frac_0p1.jsonl) → mesma
# data, mesmos hiperparâmetros, só muda o BACKEND. Compara wall-clock e PPL final.
#
# ISOLAMENTO: Unsloth fixa versões de torch/triton; instalar no venv compartilhado
# quebraria a campanha em curso. Por isso usamos um venv DEDICADO (.venv_unsloth).
# O checkpoint é salvo em 16-bit "merged" (HF padrão), avaliado pelo venv principal.
#
# Submeter da raiz do repo: ~/building-a-LLM  (só quando houver GPU livre)

set -e
REPO="$HOME/building-a-LLM"
cd "$REPO"
Q1="$REPO/trabalho-final/Q1-pretreino-continuado"
SCR="$Q1/scripts"
RES="$Q1/resultados"
BENCH="$Q1/dompi_qa.jsonl"
MODEL="Qwen/Qwen2.5-1.5B"
SUBSET="$REPO/data/train_limpo_frac_0p1.jsonl"   # mesmo do job de dose 10%
HELD_OUT="$REPO/data/held_out.jsonl"
OUTPUT_DIR="$REPO/treino/checkpoints_unsloth_0p1"
UVENV="$REPO/.venv_unsloth"
PYMAIN="$REPO/.venv/bin/python3"
mkdir -p "$RES" "$REPO/treino"

echo "=== Q1 — Unsloth benchmark (10% corpus limpo) ==="
echo "Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# 0. venv isolado p/ Unsloth (não toca no venv da campanha)
if [ ! -x "$UVENV/bin/python3" ]; then
  echo "--- Criando venv isolado $UVENV e instalando unsloth..."
  python3 -m venv "$UVENV"
  "$UVENV/bin/python3" -m pip install --quiet --upgrade pip
  "$UVENV/bin/python3" -m pip install --quiet "unsloth" "bitsandbytes"
fi
PYU="$UVENV/bin/python3"
echo "--- versões (venv unsloth):"
$PYU -c "import torch,unsloth;print('torch',torch.__version__,'| unsloth',unsloth.__version__)"

# pré-requisito: subset de 10% (criado pelo job de dose; se ausente, fatia agora)
if [ ! -f "$SUBSET" ]; then
  echo "--- subset 10% ausente; fatiando do corpus limpo..."
  TOTAL=$(wc -l < "$REPO/data/train_corpus_limpo.jsonl")
  N=$(python3 -c "print(int($TOTAL*0.1))")
  head -n "$N" "$REPO/data/train_corpus_limpo.jsonl" > "$SUBSET"
fi

# 1. treino Unsloth (cronometrado)
echo "--- [1] Treino Unsloth (full-FT, 10%)..."
T0=$(date +%s)
$PYU "$SCR/pretreino_fullft_unsloth.py" \
    --model "$MODEL" \
    --train-data "$SUBSET" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 --block-size 512 --batch-size 2 --grad-accum 8 \
    --lr 2e-6 --warmup-fraction 0.05 --gradient-checkpointing \
    --eval-every 250 --patience 3
T1=$(date +%s)
WALL=$(( (T1 - T0) / 60 ))
echo "WALL-CLOCK Unsloth (10%): ${WALL} min"

# 2. avaliação pelo venv PRINCIPAL (checkpoint é HF 16-bit padrão)
EVAL_CKPT="$OUTPUT_DIR/best"; [ -d "$EVAL_CKPT" ] || EVAL_CKPT="$OUTPUT_DIR/final"
EVAL_CKPT="$(realpath "$EVAL_CKPT")"
echo "--- [2] Avaliando $EVAL_CKPT (venv principal)"
$PYMAIN "$SCR/avaliar_modelo.py" --model "$EVAL_CKPT" \
    --held-out "$HELD_OUT" --benchmark "$BENCH" \
    --output "$RES/unsloth_0p1.json" --max-samples 1000

echo "=================================================================="
echo "COMPARAÇÃO A/B (mesmo 10% do corpus limpo):"
echo "  Unsloth   : wall=${WALL}min"
$PYMAIN -c "import json;d=json.load(open('$RES/unsloth_0p1.json'));print(f'  Unsloth PPL_dominio={d[\"perplexity\"]} CE={d[\"cross_entropy\"]} tokacc={d[\"token_accuracy\"]}')"
[ -f "$RES/dose_0p1.json" ] && $PYMAIN -c "import json;d=json.load(open('$RES/dose_0p1.json'));print(f'  Padrão  PPL_dominio={d[\"perplexity\"]} CE={d[\"cross_entropy\"]} tokacc={d[\"token_accuracy\"]}')" || echo "  (dose_0p1.json ainda não disponível p/ comparar PPL)"
echo "=== Concluído ==="
