#!/bin/bash
#SBATCH --job-name=dompi-fullft-teresina
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=treino/log_q3_fullft_teresina_%j.txt
#SBATCH --error=treino/log_q3_fullft_teresina_%j.txt

# Q3 — Full FT com AdamW 8-bit (bitsandbytes) no corpus Teresina HF (9.3M tokens).
#
# Hipótese: full FT com todos os parâmetros + lr muito conservador (2e-6) + early stopping
# por CE no held-out resolve o overfitting catastrófico observado em LoRA/QLoRA (Q2).
# LoRA intruder dimensions (arxiv:2410.21228) não se aplica: todos os parâmetros atualizados
# uniformemente, sem concentração de adaptação em ~18M params de baixo rank.
#
# Corpus: gutoportelaa/dom-pi-teresina-2025 (curated, tier A+B)
#   9.3M tokens vs 4M tokens do cluster — corpus 2× maior e mais limpo.
#
# Ref: llm2govbr AAAI 2024 (lr=2e-6, full FT Llama-2 7B gov BR);
#      ChipNeMo NVIDIA 2023 (lr=2e-6 CPT full FT);
#      job 389 validou que corpus Teresina só aumenta +8% PPL (vs ×100 em LoRA DOM-PI).

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-1.5B"
OUTPUT_DIR="treino/checkpoints_fullft_teresina"
HELD_OUT="data/teresina/held_out.jsonl"
TRAIN_DATA="data/teresina_hf/train_corpus.jsonl"
RESULTS_BASELINE="avaliacao/resultados_baseline_teresina_1.5b.json"
RESULTS_POSTREINO="avaliacao/resultados_q3_fullft_teresina.json"

echo "=== Q3 — Full FT (AdamW 8-bit) + Qwen2.5-1.5B + Corpus Teresina HF ==="
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

# Instala bitsandbytes se necessário
echo "--- Verificando bitsandbytes..."
$PYTHON -c "import bitsandbytes" 2>/dev/null || {
    echo "Instalando bitsandbytes..."
    $PYTHON -m pip install bitsandbytes --quiet
}
$PYTHON -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__} OK')"

# Instala datasets se necessário
echo "--- Verificando datasets..."
$PYTHON -c "import datasets" 2>/dev/null || {
    echo "Instalando datasets..."
    $PYTHON -m pip install datasets --quiet
}

# Passo 0: baixar corpus Teresina do HuggingFace (skip se já existir)
if [ ! -f "$TRAIN_DATA" ]; then
    echo "--- Baixando corpus Teresina do HuggingFace..."
    $PYTHON treino/preparar_corpus_teresina.py \
        --output-dir data/teresina_hf \
        --held-out "$HELD_OUT" \
        --quality-tiers A B
    echo "--- Stats do corpus baixado:"
    cat data/teresina_hf/stats.json
else
    echo "--- Corpus HF já existe ($TRAIN_DATA), pulando download."
    wc -l "$TRAIN_DATA"
fi

# Passo 1: baseline 1.5B no held-out Teresina (computa se não existir)
# Nota: resultados_baseline_teresina.json é para 0.5B (job 389). Precisamos de 1.5B aqui.
if [ ! -f "$RESULTS_BASELINE" ]; then
    echo "--- Avaliando baseline 1.5B no held-out Teresina..."
    $PYTHON avaliacao/avaliar_modelo.py \
        --model "$MODEL" \
        --held-out "$HELD_OUT" \
        --benchmark benchmark/dompi_qa.jsonl \
        --output "$RESULTS_BASELINE"
    echo "--- Baseline 1.5B Teresina:"
    cat "$RESULTS_BASELINE"
else
    echo "--- Baseline 1.5B Teresina já existe:"
    cat "$RESULTS_BASELINE"
fi

# Passo 2: treino Full FT com early stopping por CE held-out
# lr=2e-6; block=512; eval_every=200; patience=3
# ~9.100 docs treino × ~1000 tokens/doc → ~9.1M tokens → ~17.773 blocos de 512
# → 1.110 steps (bs=2, accum=8) → eval em steps 200/400/600/800/1000
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
    --eval-every 200 \
    --save-every 200 \
    --patience 3

# Passo 3: avaliação pós-treino (usa 'best' por CE held-out, cai para 'final')
echo "--- Avaliando pós-treino Full FT..."
EVAL_CKPT="$OUTPUT_DIR/best"
if [ ! -d "$EVAL_CKPT" ]; then
    EVAL_CKPT="$OUTPUT_DIR/final"
    echo "[aviso] best ausente, usando final"
fi
echo "Checkpoint avaliado: $EVAL_CKPT"

$PYTHON avaliacao/avaliar_modelo.py \
    --model "$(realpath $EVAL_CKPT)" \
    --held-out "$HELD_OUT" \
    --benchmark benchmark/dompi_qa.jsonl \
    --output "$RESULTS_POSTREINO"

# Passo 4: comparação baseline × pós-treino
echo "--- Relatório antes × depois (Full FT Teresina):"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline "$RESULTS_BASELINE" \
    --postreino "$RESULTS_POSTREINO"

# Passo 5: inferência comparativa
echo "--- Inferência comparativa Full FT vs baseline..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline_1.5b:$MODEL" "fullft_teresina:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_q3_fullft_teresina.json" \
    --max-new-tokens 200

echo ""
echo "=== Q3 Full FT Teresina concluído: $(date) ==="
echo ""
echo "Arquivos gerados:"
ls -lh "$RESULTS_POSTREINO" "avaliacao/inferencias_q3_fullft_teresina.json" 2>/dev/null || true
echo ""
echo "Interpretação:"
echo "  CE pós-treino < CE baseline → full FT + lr conservador resolve overfitting LoRA ✓"
echo "  CE pós-treino > CE baseline → overfitting persiste mesmo com full FT           ✗"
