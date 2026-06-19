#!/bin/bash
#SBATCH --job-name=q1-eval-v3
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=treino/log_q1_eval_v3_%j.txt
#SBATCH --error=treino/log_q1_eval_v3_%j.txt

# Avaliação final do checkpoint 'best' do v3 (job 489 cancelado após platô do held-out).
cd "$SLURM_SUBMIT_DIR"
PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
CKPT="treino/checkpoints_fullft_teresina_v3/best"
HELD_OUT="data/teresina/held_out.jsonl"
RESULTS_BASELINE="avaliacao/resultados_baseline_teresina_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q1_fullft_teresina_v3.json"

echo "=== Q1 v3 — avaliação final do best: $(date) ==="
$PYTHON -c "import bitsandbytes" 2>/dev/null || $PYTHON -m pip install bitsandbytes --quiet

$PYTHON avaliacao/avaliar_modelo.py --model "$(realpath $CKPT)" \
    --held-out "$HELD_OUT" --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_POSTREINO"

echo "--- Relatório antes × depois (v3):"
$PYTHON avaliacao/comparar_resultados.py --baseline "$RESULTS_BASELINE" --postreino "$RESULTS_POSTREINO"

echo "--- Inferência comparativa v3 vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "fullft_ter_v3:$(realpath $CKPT)" \
    --output "avaliacao/inferencias_q1_fullft_teresina_v3.json" --max-new-tokens 200
echo "=== concluído: $(date) ==="
