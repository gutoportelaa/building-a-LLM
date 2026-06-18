#!/bin/bash
#SBATCH --job-name=dompi-lora-0.5b
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=treino/log_q2_lora_0.5b_%j.txt
#SBATCH --error=treino/log_q2_lora_0.5b_%j.txt

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-0.5B"
OUTPUT_DIR="treino/checkpoints_lora_0.5b"
RESULTS_BASELINE="avaliacao/resultados_baseline.json"
RESULTS_POSTREINO="avaliacao/resultados_q2_lora_0.5b.json"

echo "=== Q2 — LoRA + Qwen2.5-0.5B ==="
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

# Passo 2: baseline (reutiliza se já existir)
if [ ! -f "$RESULTS_BASELINE" ]; then
    echo "--- Avaliando baseline..."
    $PYTHON avaliacao/avaliar_modelo.py \
        --model "$MODEL" \
        --held-out data/held_out.jsonl \
        --benchmark benchmark/dompi_qa.jsonl \
        --output "$RESULTS_BASELINE"
else
    echo "--- Baseline já existe, pulando avaliação."
fi

# Passo 3: inferência baseline (antes do treino)
echo "--- Inferência baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline:$MODEL" \
    --output "avaliacao/inferencias_q2_lora_0.5b_baseline.json"

# Passo 4: treino LoRA
# r=16, alpha=32 → escala de saída = alpha/r = 2.0 (padrão da literatura)
# lr=2e-5 → ~10× maior que full-FT (2e-6), alinhado ao regime DAPT.
# Raschka recomenda [1e-5, 5e-4] para LoRA; para DAPT (não instruction-tuning),
# Biderman et al. (TMLR 2024) e ChipNeMo (NVIDIA 2023) indicam LR conservador.
# lr=2e-4 causou colapso total (PPL 10→2578) — confirmado empiricamente.
echo "--- Iniciando pré-treino LoRA..."
$PYTHON treino/pretreino_lora.py \
    --model "$MODEL" \
    --method lora \
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

# Passo 5: avaliação pós-treino
echo "--- Avaliando pós-treino..."
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

# Passo 6: inferência pós-treino
echo "--- Inferência pós-treino..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline:$MODEL" "lora_0.5b_final:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q2_lora_0.5b.json"

# Passo 7: relatório
echo "--- Relatório antes x depois:"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline "$RESULTS_BASELINE" \
    --postreino "$RESULTS_POSTREINO"

echo "=== Q2 LoRA 0.5B concluído: $(date) ==="
