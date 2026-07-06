#!/bin/bash
#SBATCH --job-name=dompi-inferencias-q1
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=treino/log_inferencias_q1_%j.txt
#SBATCH --error=treino/log_inferencias_q1_%j.txt

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"

echo "=== Inferências Q1 — baseline + Job384 (intermediário) ==="
echo "Job ID: $SLURM_JOB_ID | Data: $(date)"

$PYTHON -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Verifica quais checkpoints existem
echo "--- Checkpoints disponíveis:"
ls treino/checkpoints/ 2>/dev/null || echo "[aviso] treino/checkpoints/ não encontrado"

# Geração com baseline + melhor checkpoint do Job 384
# Job 383 (lr=1e-6, safe) foi sobrescrito; incluímos apenas o que existe
if [ -d treino/checkpoints/final ]; then
    CKPT_J384="$(realpath treino/checkpoints/final)"
    LABEL_J384="j384_final_lr2e6"
elif [ -d treino/checkpoints/best ]; then
    CKPT_J384="$(realpath treino/checkpoints/best)"
    LABEL_J384="j384_best_lr2e6"
else
    echo "ERRO: nenhum checkpoint do Job 384 encontrado"
    exit 1
fi

echo "--- Usando checkpoint: $CKPT_J384"

$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline:Qwen/Qwen2.5-0.5B" "$LABEL_J384:$CKPT_J384" \
    --output avaliacao/inferencias_q1_comparativa.json \
    --max-new-tokens 200

echo "=== Inferências Q1 concluídas: $(date) ==="
