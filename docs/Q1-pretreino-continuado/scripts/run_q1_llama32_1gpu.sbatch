#!/bin/bash
#SBATCH --job-name=q1-llama32-1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=trabalho-final/Q1-pretreino-continuado/resultados/log_llama32_1gpu_%j.txt
#SBATCH --error=trabalho-final/Q1-pretreino-continuado/resultados/log_llama32_1gpu_%j.txt

# Q1 — Pré-treino continuado CROSS-FAMÍLIA: Llama-3.2-3B full-FT em 1 GPU (L4).
# Reusa o pipeline validado (pretreino_fullft.py): objetivo correto (sem duplo-shift),
# AdamW 8-bit, cosine+warmup, early-stop CE no held-out. 3B cabe numa L4 (24GB) com
# AdamW-8bit + grad checkpointing → FSDP é desnecessário neste tamanho.
# Corpus: BRUTO unificado (mesma base da canônica Qwen, p/ comparação cross-família).
# Modelo: pesos Llama-3.2-3B (Meta) via mirror ungated. Submeter da raiz: ~/building-a-LLM

set -e
REPO="$HOME/building-a-LLM"
cd "$REPO"
PYTHON="$REPO/.venv/bin/python3"
Q1="$REPO/trabalho-final/Q1-pretreino-continuado"
SCR="$Q1/scripts"
BENCH="$Q1/dompi_qa.jsonl"
RES="$Q1/resultados"
MODEL="unsloth/Llama-3.2-3B"     # pesos Meta, mirror ungated
TRAIN="$REPO/data/train_corpus.jsonl"
HELD_OUT="$REPO/data/held_out.jsonl"
OUTPUT_DIR="$REPO/treino/checkpoints_llama32_3b_dompi"
mkdir -p "$RES" "$REPO/treino"

echo "=== Q1 cross-família — Llama-3.2-3B full-FT (1 L4) ==="
echo "Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PYTHON -c "import bitsandbytes" 2>/dev/null || $PYTHON -m pip install bitsandbytes --quiet

# 1. Treino (batch 1 x accum 16 = efetivo 16, igual ao Qwen; seguro p/ 3B em 24GB)
echo "--- [1] Full-FT Llama-3.2-3B..."
$PYTHON "$SCR/pretreino_fullft.py" \
    --model "$MODEL" \
    --train-data "$TRAIN" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 --block-size 512 \
    --batch-size 1 --grad-accum 16 \
    --lr 2e-6 --warmup-fraction 0.05 --gradient-checkpointing \
    --eval-every 250 --patience 3

# 2. Avaliação antes×depois (mesmo held-out + benchmark + geração)
BEST="$OUTPUT_DIR/best"; [ -d "$BEST" ] || BEST="$OUTPUT_DIR/final"
BEST="$(realpath "$BEST")"
echo "--- [2] Avaliação baseline vs DAPT"
$PYTHON "$SCR/avaliar_modelo.py" --model "$MODEL" \
    --held-out "$HELD_OUT" --benchmark "$BENCH" \
    --output "$RES/dominio_llama32_baseline.json" --max-samples 1000
$PYTHON "$SCR/avaliar_modelo.py" --model "$BEST" \
    --held-out "$HELD_OUT" --benchmark "$BENCH" \
    --output "$RES/dominio_llama32_dapt.json" --max-samples 1000
$PYTHON "$SCR/avaliar_geracao.py" --model "$MODEL" \
    --benchmark "$BENCH" --output "$RES/geracao_llama32_baseline.json"
$PYTHON "$SCR/avaliar_geracao.py" --model "$BEST" \
    --benchmark "$BENCH" --output "$RES/geracao_llama32_dapt.json"

echo "=================================================================="
echo "RESULTADO Llama-3.2-3B (antes -> depois) | comparar com Qwen2.5-1.5B:"
$PYTHON -c "import json;b=json.load(open('$RES/dominio_llama32_baseline.json'));d=json.load(open('$RES/dominio_llama32_dapt.json'));print(f'  PPL {b[\"perplexity\"]} -> {d[\"perplexity\"]} | CE {b[\"cross_entropy\"]} -> {d[\"cross_entropy\"]} | tokacc {b[\"token_accuracy\"]} -> {d[\"token_accuracy\"]}')"
echo "=== Concluído ==="
