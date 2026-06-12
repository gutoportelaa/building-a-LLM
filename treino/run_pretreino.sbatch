#!/bin/bash
#SBATCH --job-name=dompi-pretreino
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=treino/log_slurm_%j.txt
#SBATCH --error=treino/log_slurm_%j.txt

set -e
cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"

echo "=== Iniciando pré-treino DOM-PI ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Passo 1: gerar held-out/train a partir do HF Hub (sem precisar de Parquets locais)
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
echo "--- Iniciando pré-treino..."
$PYTHON treino/pretreino_continuado.py \
    --model Qwen/Qwen2.5-0.5B \
    --train-data data/train_corpus.jsonl \
    --output-dir treino/checkpoints \
    --epochs 2 \
    --block-size 1024 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-5

# Passo 4: avaliação pós-treino
echo "--- Avaliando pós-treino..."
$PYTHON avaliacao/avaliar_modelo.py \
    --model treino/checkpoints/final \
    --held-out data/held_out.jsonl \
    --benchmark benchmark/dompi_qa.jsonl \
    --output avaliacao/resultados_postreino.json

# Passo 5: relatório
echo "--- Relatório antes x depois:"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline avaliacao/resultados_baseline.json \
    --postreino avaliacao/resultados_postreino.json

echo "=== Concluído: $(date) ==="
