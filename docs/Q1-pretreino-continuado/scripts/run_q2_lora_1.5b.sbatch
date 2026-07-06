#!/bin/bash
#SBATCH --job-name=dompi-lora-1.5b
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=treino/log_q2_lora_1.5b_%j.txt
#SBATCH --error=treino/log_q2_lora_1.5b_%j.txt

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_lora_1.5b"
RESULTS_BASELINE="avaliacao/resultados_baseline_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q2_lora_1.5b.json"

echo "=== Q2 — LoRA + Qwen2.5-1.5B ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[aviso] nvidia-smi indisponível"

$PYTHON -c "
import torch, sys
print(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('ERRO: CUDA não disponível — abortando')
    sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"

# Passo 1: held-out (reutiliza se já existir)
if [ ! -f data/held_out.jsonl ]; then
    echo "--- Preparando held-out..."
    $PYTHON avaliacao/preparar_held_out.py --from-hub --output-dir data
fi

# Passo 2: baseline 1.5B
echo "--- Avaliando baseline 1.5B..."
$PYTHON avaliacao/avaliar_modelo.py \
    --model "$MODEL" \
    --held-out data/held_out.jsonl \
    --benchmark benchmark/dompi_qa.jsonl \
    --output "$RESULTS_BASELINE"

# Passo 3: inferência baseline
echo "--- Inferência baseline 1.5B..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" \
    --output "avaliacao/inferencias_q2_lora_1.5b_baseline.json"

# Passo 4: treino LoRA
# 1.5B tem mais parâmetros → passos mais lentos → lr um pouco menor (2e-4)
# gradient_checkpointing para reduzir consumo de memória no modelo maior
echo "--- Iniciando pré-treino LoRA (1.5B)..."
$PYTHON treino/pretreino_lora.py \
    --model "$MODEL" \
    --method lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --train-data data/train_corpus.jsonl \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --block-size 1024 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-4 \
    --warmup-fraction 0.05 \
    --gradient-checkpointing \
    --log-every 20 \
    --save-every 500

# Passo 5: avaliação pós-treino
echo "--- Avaliando pós-treino 1.5B..."
EVAL_CKPT="$OUTPUT_DIR/final"
if [ ! -d "$EVAL_CKPT" ]; then
    EVAL_CKPT="$OUTPUT_DIR/best"
    echo "[aviso] final ausente, usando best"
fi
$PYTHON avaliacao/avaliar_modelo.py \
    --model "$(realpath $EVAL_CKPT)" \
    --held-out data/held_out.jsonl \
    --benchmark benchmark/dompi_qa.jsonl \
    --output "$RESULTS_POSTREINO"

# Passo 6: inferência comparativa
echo "--- Inferência pós-treino 1.5B..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "lora_1.5b_final:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q2_lora_1.5b.json"

# Passo 7: relatório
echo "--- Relatório antes x depois (1.5B):"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline "$RESULTS_BASELINE" \
    --postreino "$RESULTS_POSTREINO"

echo "=== Q2 LoRA 1.5B concluído: $(date) ==="
