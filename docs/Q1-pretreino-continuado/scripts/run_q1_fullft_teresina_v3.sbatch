#!/bin/bash
#SBATCH --job-name=q1-fullft-ter-v3
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --output=treino/log_q1_fullft_teresina_v3_%j.txt
#SBATCH --error=treino/log_q1_fullft_teresina_v3_%j.txt

# Q1 — Full FT Teresina v3 (CORREÇÃO DO OBJETIVO DE TREINO).
#
# Bug encontrado em pretreino_fullft.py (PackedTextDataset): labels eram
# pré-deslocados (block[1:]) E o modelo HF desloca internamente → DUPLO shift →
# o treino otimizava "prever 2 tokens à frente" (objetivo errado) e a eval_held_out
# reportava PPL inflada (~47000 no baseline vs 6,91 real do avaliar_modelo.py).
# Afetou TODAS as rodadas anteriores (v1/v2/unificado) — só não colapsaram porque
# lr=2e-6 é minúsculo. Corrigido: input_ids e labels alinhados (shift só interno).
#
# Com o objetivo certo: (a) early stopping por CE no held-out passa a funcionar de
# verdade; (b) o treino aponta na direção correta — chance real de PPL < baseline.
#
# v2 → v3:
#   objetivo de treino: DUPLO shift (bug) → next-token correto
#   --epochs 3 → 2          (early stopping real protege contra overfit)
#   --freeze-embeddings     → REMOVIDO (isolar o efeito da correção; v1 não usava)
#   lr=2e-6, patience=3, eval_every=200 mantidos
#
# Corpus: gutoportelaa/dom-pi-teresina-2025 (curated, tier A+B, ~9.3M tokens).

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_fullft_teresina_v3"
HELD_OUT="data/teresina/held_out.jsonl"
TRAIN_DATA="data/teresina_hf/train_corpus.jsonl"
RESULTS_BASELINE="avaliacao/resultados_baseline_teresina_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q1_fullft_teresina_v3.json"

echo "=== Q1 — Full FT Teresina v3 (objetivo corrigido + 2 épocas + early stopping real) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[aviso] nvidia-smi indisponível"

$PYTHON -c "
import torch, sys
print(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('ERRO: CUDA não disponível — abortando'); sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"

echo "--- Verificando bitsandbytes..."
$PYTHON -c "import bitsandbytes" 2>/dev/null || $PYTHON -m pip install bitsandbytes --quiet
$PYTHON -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__} OK')"

echo "--- Verificando datasets..."
$PYTHON -c "import datasets" 2>/dev/null || $PYTHON -m pip install datasets --quiet

# Passo 0: corpus Teresina (skip se já existir)
if [ ! -f "$TRAIN_DATA" ]; then
    echo "--- Baixando corpus Teresina do HuggingFace..."
    $PYTHON treino/preparar_corpus_teresina.py \
        --output-dir data/teresina_hf \
        --held-out "$HELD_OUT" \
        --quality-tiers A B
    cat data/teresina_hf/stats.json
else
    echo "--- Corpus HF já existe ($TRAIN_DATA):"; wc -l "$TRAIN_DATA"
fi

# Passo 1: baseline 1.5B no held-out Teresina (reusa se existir)
if [ ! -f "$RESULTS_BASELINE" ]; then
    echo "--- Avaliando baseline 1.5B no held-out Teresina..."
    $PYTHON avaliacao/avaliar_modelo.py --model "$MODEL" \
        --held-out "$HELD_OUT" --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_BASELINE"
fi
echo "--- Baseline 1.5B Teresina:"; cat "$RESULTS_BASELINE"

# Passo 2: Full FT v3 — objetivo corrigido, 2 épocas, early stopping real por CE
# Sanidade: com o bug corrigido a eval_held_out baseline DEVE ficar ~6,9 (não ~47000)
echo "--- Iniciando Full FT v3 (lr=2e-6, 2 épocas, objetivo corrigido, early stopping)..."
$PYTHON treino/pretreino_fullft.py \
    --model "$MODEL" \
    --train-data "$TRAIN_DATA" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 2 \
    --block-size 512 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-6 \
    --warmup-fraction 0.05 \
    --gradient-checkpointing \
    --log-every 20 \
    --eval-every 200 \
    --save-every 200 \
    --patience 3

# Passo 3: avaliação pós-treino (best por CE held-out, cai para final)
EVAL_CKPT="$OUTPUT_DIR/best"
[ -d "$EVAL_CKPT" ] || EVAL_CKPT="$OUTPUT_DIR/final"
echo "--- Avaliando pós-treino: $EVAL_CKPT"
$PYTHON avaliacao/avaliar_modelo.py --model "$(realpath $EVAL_CKPT)" \
    --held-out "$HELD_OUT" --benchmark benchmark/dompi_qa.jsonl --output "$RESULTS_POSTREINO"

echo "--- Relatório antes × depois (Full FT Teresina v3):"
$PYTHON avaliacao/comparar_resultados.py --baseline "$RESULTS_BASELINE" --postreino "$RESULTS_POSTREINO"

echo "--- Inferência comparativa v3 vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "fullft_ter_v3:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q1_fullft_teresina_v3.json" --max-new-tokens 200

echo "=== Q1 Full FT Teresina v3 concluído: $(date) ==="
