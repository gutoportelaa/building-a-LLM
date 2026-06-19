#!/bin/bash
#SBATCH --job-name=q1-qlora-unif-v2
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=treino/log_q1_qlora_unificado_v2_%j.txt
#SBATCH --error=treino/log_q1_qlora_unificado_v2_%j.txt

# Q1 — QLoRA no dataset COMPLETO (unificado), OBJETIVO DE TREINO CORRIGIDO.
# Re-execução do job 421 (QLoRA DOM-PI, +4,4%) sem o bug de duplo shift do
# pretreino_lora.py. Mesmos hiperparâmetros (lr=2e-5, rank16, NF4, 1 época).
cd "$SLURM_SUBMIT_DIR"
PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_qlora_1.5b_v2"
RESULTS_BASELINE="avaliacao/resultados_baseline_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q1_qlora_unificado_v2.json"

echo "=== Q1 — QLoRA unificado v2 (objetivo corrigido): $(date) ==="
$PYTHON -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo 'sem CUDA'; exit 1; }
$PYTHON -c "import bitsandbytes" 2>/dev/null || $PYTHON -m pip install bitsandbytes --quiet

if [ ! -f "$RESULTS_BASELINE" ]; then
    $PYTHON avaliacao/avaliar_modelo.py --model "$MODEL" \
        --held-out data/held_out.jsonl --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_BASELINE"
fi
echo "--- Baseline:"; cat "$RESULTS_BASELINE"

echo "--- Treino QLoRA unificado v2 (objetivo corrigido)..."
$PYTHON treino/pretreino_lora.py \
    --model "$MODEL" \
    --method qlora \
    --lora-r 16 \
    --lora-alpha 32 \
    --train-data data/train_corpus.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --block-size 1024 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-5 \
    --warmup-fraction 0.05 \
    --log-every 20 \
    --save-every 500

EVAL_CKPT="$OUTPUT_DIR/final"
[ -d "$EVAL_CKPT" ] || EVAL_CKPT="$OUTPUT_DIR/best"
echo "--- Avaliando pós-treino: $EVAL_CKPT"
$PYTHON avaliacao/avaliar_modelo.py --model "$(realpath $EVAL_CKPT)" \
    --held-out data/held_out.jsonl --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_POSTREINO"

echo "--- Relatório antes × depois (QLoRA unificado v2):"
$PYTHON avaliacao/comparar_resultados.py --baseline "$RESULTS_BASELINE" --postreino "$RESULTS_POSTREINO"

echo "--- Inferência comparativa QLoRA v2 vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "qlora_unif_v2:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q1_qlora_unificado_v2.json"
echo "=== concluído: $(date) ==="
