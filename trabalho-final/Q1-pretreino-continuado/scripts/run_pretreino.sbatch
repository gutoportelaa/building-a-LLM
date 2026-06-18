#!/bin/bash
#SBATCH --job-name=dompi-pretreino
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=treino/log_slurm_%j.txt
#SBATCH --error=treino/log_slurm_%j.txt

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"

echo "=== Iniciando pré-treino DOM-PI ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[aviso] nvidia-smi indisponível no nó"

# Confirma que torch enxerga a GPU antes de prosseguir
$PYTHON -c "
import torch, sys
print(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('ERRO: CUDA não disponível — abortando')
    sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"

# Passo 1: gerar held-out/train a partir do HF Hub
echo "--- Preparando held-out..."
$PYTHON avaliacao/preparar_held_out.py --from-hub --output-dir data

# Passo 2: baseline ANTES do treino
echo "--- Avaliando baseline..."
$PYTHON avaliacao/avaliar_modelo.py \
    --model Qwen/Qwen2.5-0.5B \
    --held-out data/held_out.jsonl \
    --benchmark benchmark/dompi_qa.jsonl \
    --output avaliacao/resultados_baseline.json

# Passo 3: pré-treino continuado
# LR schedule manual (Apêndice D do Raschka): warmup linear 5% → cosine decay
# peak_lr=1e-6 (5x menor que tentativa anterior) para reduzir esquecimento catastrófico
# 1 época (Raschka: "common to train for only one epoch" em corpus grande)
echo "--- Iniciando pré-treino..."
$PYTHON treino/pretreino_continuado.py \
    --model Qwen/Qwen2.5-0.5B \
    --train-data data/train_corpus.jsonl \
    --output-dir treino/checkpoints \
    --epochs 1 \
    --block-size 1024 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-6 \
    --warmup-fraction 0.05

# Passo 4: avaliação pós-treino (usa final se existir, senão best)
echo "--- Avaliando pós-treino..."
EVAL_CKPT="treino/checkpoints/final"
if [ ! -d "$EVAL_CKPT" ]; then
    EVAL_CKPT="treino/checkpoints/best"
    echo "[aviso] checkpoint final ausente; usando $EVAL_CKPT"
fi
$PYTHON avaliacao/avaliar_modelo.py \
    --model "$(realpath $EVAL_CKPT)" \
    --held-out data/held_out.jsonl \
    --benchmark benchmark/dompi_qa.jsonl \
    --output avaliacao/resultados_postreino.json

# Passo 5: relatório
echo "--- Relatório antes x depois:"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline avaliacao/resultados_baseline.json \
    --postreino avaliacao/resultados_postreino.json

echo "=== Concluído: $(date) ==="
