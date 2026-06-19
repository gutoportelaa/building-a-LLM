#!/bin/bash
#SBATCH --job-name=q1-fullft-unif-v2
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=treino/log_q1_fullft_unificado_v2_%j.txt
#SBATCH --error=treino/log_q1_fullft_unificado_v2_%j.txt

# Q1 — Full FT no dataset COMPLETO (unificado), OBJETIVO DE TREINO CORRIGIDO.
#
# Re-execução da "resposta canônica" do enunciado (dataset completo DOM-PI) agora
# SEM o bug do duplo shift (que contaminava o job 487 → +121%). Objetivo: obter um
# número justo para a resposta canônica; os +121% anteriores eram artefato do bug.
# Sem freeze (comparável ao 487). Early stopping confiável (eval corrigida + --min-delta).

cd "$SLURM_SUBMIT_DIR"
PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_fullft_unificado_v2"
HELD_OUT="data/held_out.jsonl"
TRAIN_DATA="data/train_corpus.jsonl"
RESULTS_BASELINE="avaliacao/resultados_baseline_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_fullft_unificado_v2.json"

echo "=== Q1 — Full FT unificado v2 (objetivo corrigido): $(date) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
$PYTHON -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'sem CUDA'; exit 1; }
$PYTHON -c "import bitsandbytes" 2>/dev/null || $PYTHON -m pip install bitsandbytes --quiet

# Baseline unificado (reusa se existir)
if [ ! -f "$RESULTS_BASELINE" ]; then
    echo "--- Avaliando baseline 1.5B no held-out unificado..."
    $PYTHON avaliacao/avaliar_modelo.py --model "$MODEL" \
        --held-out "$HELD_OUT" --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_BASELINE"
fi
echo "--- Baseline:"; cat "$RESULTS_BASELINE"
# Sanidade esperada (objetivo corrigido): eval_held_out baseline interno ~CE 2,3 / PPL ~10 (NÃO ~47000)

echo "--- Treino Full FT unificado v2 (lr=2e-6, 1 época, objetivo corrigido, early stopping)..."
$PYTHON treino/pretreino_fullft.py \
    --model "$MODEL" \
    --train-data "$TRAIN_DATA" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --block-size 512 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-6 \
    --warmup-fraction 0.05 \
    --gradient-checkpointing \
    --log-every 20 \
    --eval-every 500 \
    --save-every 500 \
    --patience 3 \
    --min-delta 1e-3

EVAL_CKPT="$OUTPUT_DIR/best"
[ -d "$EVAL_CKPT" ] || EVAL_CKPT="$OUTPUT_DIR/final"
echo "--- Avaliando pós-treino: $EVAL_CKPT"
$PYTHON avaliacao/avaliar_modelo.py --model "$(realpath $EVAL_CKPT)" \
    --held-out "$HELD_OUT" --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_POSTREINO"

echo "--- Relatório antes × depois (unificado v2):"
$PYTHON avaliacao/comparar_resultados.py --baseline "$RESULTS_BASELINE" --postreino "$RESULTS_POSTREINO"

echo "--- Inferência comparativa unificado v2 vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "fullft_unif_v2:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q1_fullft_unificado_v2.json" --max-new-tokens 200
echo "=== concluído: $(date) ==="
