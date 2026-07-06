#!/bin/bash
#SBATCH --job-name=dompi-lora-1.5b-v2
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=treino/log_q2_lora_1.5b_v2_%j.txt
#SBATCH --error=treino/log_q2_lora_1.5b_v2_%j.txt

# LoRA 1.5B v2 — lr corrigido de 2e-4 (colapso, PPL 10→2578) para 2e-5 (DAPT).
# Ref: Biderman et al. "LoRA Learns Less and Forgets Less" TMLR 2024 (arxiv:2405.09673);
#      ChipNeMo NVIDIA 2023 (arxiv:2311.00176); Raschka lr range [1e-5,5e-4] para DAPT.
# Output em checkpoints_lora_1.5b_v2 (preserva checkpoint com falha para comparação).

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_lora_1.5b_v2"
RESULTS_BASELINE="avaliacao/resultados_baseline_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q2_lora_1.5b.json"

echo "=== Q2 — LoRA + Qwen2.5-1.5B v2 (lr=2e-5) ==="
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

# Passo 2: baseline 1.5B (reutiliza se já existir do job 396)
if [ ! -f "$RESULTS_BASELINE" ]; then
    echo "--- Avaliando baseline 1.5B..."
    $PYTHON avaliacao/avaliar_modelo.py \
        --model "$MODEL" \
        --held-out data/held_out.jsonl \
        --benchmark benchmark/dompi_qa.jsonl \
        --output "$RESULTS_BASELINE"
else
    echo "--- Baseline 1.5B já existe, pulando."
fi

# Passo 3: treino LoRA — lr=2e-5 (DAPT conservador)
echo "--- Iniciando pré-treino LoRA 1.5B v2 (lr=2e-5)..."
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
    --lr 2e-5 \
    --warmup-fraction 0.05 \
    --gradient-checkpointing \
    --log-every 20 \
    --save-every 500

# Passo 4: avaliação pós-treino
echo "--- Avaliando pós-treino LoRA 1.5B v2..."
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

# Passo 5: inferência comparativa
echo "--- Inferência comparativa LoRA 1.5B v2 vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "lora_1.5b_final:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q2_lora_1.5b.json"

# Passo 6: relatório
echo "--- Relatório antes x depois (LoRA 1.5B v2):"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline "$RESULTS_BASELINE" \
    --postreino "$RESULTS_POSTREINO"

echo "=== Q2 LoRA 1.5B v2 concluído: $(date) ==="
