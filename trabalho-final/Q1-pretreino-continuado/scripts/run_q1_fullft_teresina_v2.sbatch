#!/bin/bash
#SBATCH --job-name=q1-fullft-ter-v2
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=treino/log_q1_fullft_teresina_v2_%j.txt
#SBATCH --error=treino/log_q1_fullft_teresina_v2_%j.txt

# Q1 — Full FT Teresina v2 (recipe DAPT tokens-mínimos, arxiv:2507.02964).
#
# Contexto: full FT no corpus UNIFICADO (job 487) foi catastrófico (+121% PPL);
# full FT Teresina v1 (job 429, 1 época, sem freeze) foi viável (+27%). A literatura
# de DAPT com poucos tokens (50-119M) mostra bom resultado com corpus CURADO +
# embeddings CONGELADAS + 2-3 épocas + lr conservador. v2 aplica esse recipe:
#
#   v1 → v2:
#     --epochs 1     → --epochs 3        (todas as 8 tentativas usaram 1 época)
#     (sem freeze)   → --freeze-embeddings (preserva conhecimento geral)
#     lr=2e-6, early stopping por CE held-out (patience=3) mantidos
#
# Replay 5% geral pt-BR: DEFERIDO (precisa de fonte de corpus geral no cluster).
# Freeze de embeddings já cobre boa parte da retenção de conhecimento geral; se a
# degradação no benchmark de domínio/geral persistir, adicionamos replay em v3.
#
# Corpus: gutoportelaa/dom-pi-teresina-2025 (curated, tier A+B, ~9.3M tokens).

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_fullft_teresina_v2"
HELD_OUT="data/teresina/held_out.jsonl"
TRAIN_DATA="data/teresina_hf/train_corpus.jsonl"
RESULTS_BASELINE="avaliacao/resultados_baseline_teresina_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q1_fullft_teresina_v2.json"

echo "=== Q1 — Full FT Teresina v2 (3 épocas + embeddings congeladas) ==="
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

# Passo 2: Full FT v2 — 3 épocas, embeddings congeladas, early stopping por CE
echo "--- Iniciando Full FT v2 (lr=2e-6, 3 épocas, freeze embeddings, early stopping)..."
$PYTHON treino/pretreino_fullft.py \
    --model "$MODEL" \
    --train-data "$TRAIN_DATA" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 3 \
    --block-size 512 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-6 \
    --warmup-fraction 0.05 \
    --gradient-checkpointing \
    --freeze-embeddings \
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

echo "--- Relatório antes × depois (Full FT Teresina v2):"
$PYTHON avaliacao/comparar_resultados.py --baseline "$RESULTS_BASELINE" --postreino "$RESULTS_POSTREINO"

echo "--- Inferência comparativa v2 vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "fullft_ter_v2:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q1_fullft_teresina_v2.json" --max-new-tokens 200

echo "=== Q1 Full FT Teresina v2 concluído: $(date) ==="
