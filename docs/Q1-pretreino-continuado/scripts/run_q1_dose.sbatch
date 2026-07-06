#!/bin/bash
#SBATCH --job-name=q1-dose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=trabalho-final/Q1-pretreino-continuado/resultados/log_dose_%j.txt
#SBATCH --error=trabalho-final/Q1-pretreino-continuado/resultados/log_dose_%j.txt

# Q1 — Eixo B: ponto da CURVA DOSE-RESPOSTA (PPL x tokens de DAPT).
# Treina full-FT numa FRAÇÃO do corpus LIMPO (mesmo do Eixo A) e avalia no held-out
# FIXO. O ponto de 100% = job do Eixo A (run_q1_fullft_limpo). Aqui geramos os
# pontos intermediários para mostrar o ganho ~log(N_tokens).
#
# Submeter da raiz do repo com FRAC:
#   sbatch --export=ALL,FRAC=0.1 trabalho-final/Q1-pretreino-continuado/scripts/run_q1_dose.sbatch
#   sbatch --export=ALL,FRAC=0.3 ...

set -e
FRAC="${FRAC:-0.1}"
TAG="${FRAC/./p}"   # 0.1 -> 0p1 (nome de arquivo)
REPO="$HOME/building-a-LLM"
cd "$REPO"
PYTHON="$REPO/.venv/bin/python3"
Q1="$REPO/trabalho-final/Q1-pretreino-continuado"
SCR="$Q1/scripts"
BENCH="$Q1/dompi_qa.jsonl"
RES="$Q1/resultados"
MODEL="Qwen/Qwen2.5-1.5B"
TRAIN_LIMPO="$REPO/data/train_corpus_limpo.jsonl"
HELD_OUT="$REPO/data/held_out.jsonl"
SUBSET="$REPO/data/train_limpo_frac_${TAG}.jsonl"
OUTPUT_DIR="$REPO/treino/checkpoints_dose_${TAG}"
mkdir -p "$RES" "$REPO/treino"

echo "=== Q1 Eixo B — dose-resposta FRAC=$FRAC ==="
echo "Node: $SLURMD_NODENAME | Data: $(date)"

# 1. fatia o corpus limpo (primeiros N docs = FRAC do total)
TOTAL=$(wc -l < "$TRAIN_LIMPO")
N=$(python3 -c "print(int($TOTAL*$FRAC))")
echo "--- [1] Subconjunto: $N de $TOTAL docs (FRAC=$FRAC)"
head -n "$N" "$TRAIN_LIMPO" > "$SUBSET"

# 2. DAPT full-FT no subconjunto (mesmos hiperparâmetros do Eixo A)
echo "--- [2] Treino full-FT no subconjunto..."
$PYTHON "$SCR/pretreino_fullft.py" \
    --model "$MODEL" \
    --train-data "$SUBSET" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --block-size 512 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-6 \
    --warmup-fraction 0.05 \
    --gradient-checkpointing \
    --eval-every 250 \
    --patience 3

# 3. avalia no held-out fixo + benchmark
EVAL_CKPT="$OUTPUT_DIR/best"; [ -d "$EVAL_CKPT" ] || EVAL_CKPT="$OUTPUT_DIR/final"
EVAL_CKPT="$(realpath "$EVAL_CKPT")"
echo "--- [3] Avaliando $EVAL_CKPT"
$PYTHON "$SCR/avaliar_modelo.py" --model "$EVAL_CKPT" \
    --held-out "$HELD_OUT" --benchmark "$BENCH" \
    --output "$RES/dose_${TAG}.json" --max-samples 1000

# anota tokens nominais (chars do subset / ~4.3) para o eixo-x da curva
$PYTHON - "$SUBSET" "$RES/dose_${TAG}.json" "$FRAC" <<'PY'
import json, sys
subset, resj, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
chars = sum(len(json.loads(l).get("texto","")) for l in open(subset) if l.strip())
d = json.load(open(resj))
d["frac"] = frac
d["chars"] = chars
d["tokens_nominais"] = int(chars / 4.3)
json.dump(d, open(resj, "w"), ensure_ascii=False, indent=2)
print(f"FRAC={frac} tokens~{d['tokens_nominais']} PPL={d['perplexity']}")
PY
echo "=== Concluído FRAC=$FRAC ==="
