#!/bin/bash
#SBATCH --job-name=dompi-qlora-1.5b
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=treino/log_q2_qlora_1.5b_%j.txt
#SBATCH --error=treino/log_q2_qlora_1.5b_%j.txt

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_qlora_1.5b"
RESULTS_BASELINE="avaliacao/resultados_baseline_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q2_qlora_1.5b.json"

echo "=== Q2 — QLoRA (4-bit NF4) + Qwen2.5-1.5B ==="
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

# Instala bitsandbytes se necessário (pip pode não existir como binário — usar python -m pip)
echo "--- Verificando bitsandbytes..."
$PYTHON -c "import bitsandbytes" 2>/dev/null || {
    echo "bitsandbytes ausente — instalando via python -m pip..."
    $PYTHON -m ensurepip --upgrade 2>/dev/null || true
    $PYTHON -m pip install bitsandbytes --quiet
}
$PYTHON -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__} OK')"

# Passo 1: held-out (reutiliza se já existir)
if [ ! -f data/held_out.jsonl ]; then
    echo "--- Preparando held-out..."
    $PYTHON avaliacao/preparar_held_out.py --from-hub --output-dir data
fi

# Passo 2: baseline 1.5B (reutiliza se já existir do job LoRA 1.5B)
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

# Passo 3: inferência baseline (reutiliza se já existir)
if [ ! -f avaliacao/inferencias_q2_lora_1.5b_baseline.json ]; then
    echo "--- Inferência baseline 1.5B..."
    $PYTHON avaliacao/inferencia_multi.py \
        --models "baseline_1.5b:$MODEL" \
        --output "avaliacao/inferencias_q2_qlora_1.5b_baseline.json"
fi

# Passo 4: treino QLoRA
# 4-bit NF4 + double quantization reduz o modelo base p/ ~600MB (vs 3GB bf16)
# Permite batch_size=4 no L4 com o 1.5B model
# lr=2e-5 — DAPT requer LR ~10× full-FT (2e-6). lr=2e-4 causou PPL 10→2578 (colapso).
# Ref: Biderman et al. TMLR 2024; ChipNeMo NVIDIA 2023; Raschka range [1e-5,5e-4].
# PagedAdamW32bit: gerencia paginação de estados do otimizador na GPU
echo "--- Iniciando pré-treino QLoRA (1.5B, 4-bit NF4)..."
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

# Passo 5: avaliação pós-treino
# Nota: o modelo merged (final/best) está em bf16; a avaliação não precisa de bitsandbytes
echo "--- Avaliando pós-treino QLoRA..."
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

# Passo 6: inferência comparativa (baseline + qlora)
echo "--- Inferência comparativa QLoRA vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "qlora_1.5b_final:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q2_qlora_1.5b.json"

# Passo 7: relatório
echo "--- Relatório antes x depois (QLoRA 1.5B):"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline "$RESULTS_BASELINE" \
    --postreino "$RESULTS_POSTREINO"

echo "=== Q2 QLoRA 1.5B concluído: $(date) ==="
