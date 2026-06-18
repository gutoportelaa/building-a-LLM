#!/bin/bash
#SBATCH --job-name=dompi-fullft-unif
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=treino/log_fullft_unificado_%j.txt
#SBATCH --error=treino/log_fullft_unificado_%j.txt

# Q1 (resposta canônica) — Full FT no corpus UNIFICADO diariosPrefeituras.
#
# Motivação: "pré-treinamento continuado" é, por definição, treino de TODOS os
# parâmetros. LoRA/QLoRA são PEFT. O Full FT bom (job 429) foi só no subset
# Teresina; no unificado só houve Full FT ingênuo (381-384, sem early stopping →
# colapso). Esta rodada usa a receita validada (AdamW-8bit + early stopping por
# CE no held-out + lr=2e-6) no corpus que o enunciado pede, permitindo comparação
# maçã-com-maçã com o QLoRA (mesmo corpus, mesmo held-out).

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_fullft_unificado"
HELD_OUT="data/held_out.jsonl"
TRAIN_DATA="data/train_corpus.jsonl"
RESULTS_BASELINE="avaliacao/resultados_baseline_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_fullft_unificado.json"

echo "=== Q1 — Full FT (AdamW 8-bit) + Qwen2.5-1.5B + Corpus UNIFICADO ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[aviso] nvidia-smi indisponível"

$PYTHON -c "
import torch, sys
print(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('ERRO: CUDA não disponível'); sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"

echo "--- Verificando bitsandbytes..."
$PYTHON -c "import bitsandbytes" 2>/dev/null || $PYTHON -m pip install bitsandbytes --quiet
$PYTHON -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__} OK')"

# Baseline 1.5B no held-out unificado (reusa se existir)
if [ ! -f "$RESULTS_BASELINE" ]; then
    echo "--- Avaliando baseline 1.5B no held-out unificado..."
    $PYTHON avaliacao/avaliar_modelo.py --model "$MODEL" \
        --held-out "$HELD_OUT" --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_BASELINE"
fi
echo "--- Baseline 1.5B unificado:"; cat "$RESULTS_BASELINE"

# Treino Full FT com early stopping por CE no held-out
echo "--- Iniciando Full FT (lr=2e-6, AdamW 8-bit, early stopping)..."
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
    --eval-every 250 \
    --save-every 250 \
    --patience 3

# Avaliação pós-treino (best por CE held-out, cai para final)
EVAL_CKPT="$OUTPUT_DIR/best"
[ -d "$EVAL_CKPT" ] || EVAL_CKPT="$OUTPUT_DIR/final"
echo "--- Avaliando pós-treino: $EVAL_CKPT"
$PYTHON avaliacao/avaliar_modelo.py --model "$(realpath $EVAL_CKPT)" \
    --held-out "$HELD_OUT" --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_POSTREINO"

echo "--- Relatório antes × depois (Full FT unificado):"
$PYTHON avaliacao/comparar_resultados.py --baseline "$RESULTS_BASELINE" --postreino "$RESULTS_POSTREINO"

echo "--- Inferência comparativa Full FT unificado vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "fullft_unif:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_fullft_unificado.json" --max-new-tokens 200

echo "=== Q1 Full FT unificado concluído: $(date) ==="
