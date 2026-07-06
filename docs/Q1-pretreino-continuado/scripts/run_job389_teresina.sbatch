#!/bin/bash
#SBATCH --job-name=dompi-teresina-lr2e6
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=treino/log_job389_teresina_%j.txt
#SBATCH --error=treino/log_job389_teresina_%j.txt

# Job 389 — Validação da hipótese de qualidade de corpus
#
# Questão: os problemas de esquecimento catastrófico nos Jobs 381/383/384 foram
# causados pelo ruído OCR do corpus geral, ou pelo LR/configuração de treino?
#
# Controle: mesmos hiperparâmetros do Job 384 (lr=2e-6, warmup=5%, 1 época,
# batch=4, grad_accum=4) — única variável diferente é o corpus.
#
# Corpus: dom-pi-teresina-2025 (config curated, Tier A+B)
#   - 9.577 docs vs 54.368 no corpus geral
#   - Texto limpo (texto estruturado, sem OCR corrompido)
#   - Loss inicial esperada: ~3–4 (vs 9.98 no corpus geral)

cd "$SLURM_SUBMIT_DIR"

PYTHON=".venv/bin/python3"
MODEL="Qwen/Qwen2.5-0.5B"
HF_REPO="gutoportelaa/dom-pi-teresina-2025"
OUTPUT_DIR="treino/checkpoints_teresina"
DATA_DIR="data/teresina"
RESULTS_BASELINE="avaliacao/resultados_baseline.json"
RESULTS_POSTREINO="avaliacao/resultados_job389_teresina.json"

echo "=== Job 389 — Full-FT Teresina (corpus limpo, lr=2e-6) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Data: $(date)"
echo "Hipótese: corpus OCR corrompido é a causa raiz do esquecimento catastrófico."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[aviso] nvidia-smi indisponível"

$PYTHON -c "
import torch, sys
print(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('ERRO: CUDA não disponível — abortando')
    sys.exit(1)
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
"

# Passo 1: preparar held-out e corpus de treino do dataset de Teresina
# Mesmo script do Q1 mas apontando para o repo de Teresina e diretório separado
echo "--- Preparando held-out (Teresina)..."
$PYTHON avaliacao/preparar_held_out.py \
    --from-hub \
    --hf-repo "$HF_REPO" \
    --output-dir "$DATA_DIR" \
    --held-out-frac 0.05 \
    --seed 42 \
    --tiers "A,B"

echo "--- Stats do corpus Teresina:"
wc -l "$DATA_DIR/held_out.jsonl" "$DATA_DIR/train_corpus.jsonl"

# Passo 2: baseline (reutiliza resultados_baseline.json já existente — mesmo modelo, seed fixo)
if [ ! -f "$RESULTS_BASELINE" ]; then
    echo "--- Avaliando baseline no held-out de Teresina..."
    $PYTHON avaliacao/avaliar_modelo.py \
        --model "$MODEL" \
        --held-out "$DATA_DIR/held_out.jsonl" \
        --benchmark benchmark/dompi_qa.jsonl \
        --output "$RESULTS_BASELINE"
else
    echo "--- Baseline já existe ($RESULTS_BASELINE), reutilizando."
    echo "    (held-out diferente: Teresina; baseline do modelo base é invariante)"
fi

# Avalia baseline também no held-out específico de Teresina
echo "--- Avaliando baseline no held-out Teresina (referência local)..."
$PYTHON avaliacao/avaliar_modelo.py \
    --model "$MODEL" \
    --held-out "$DATA_DIR/held_out.jsonl" \
    --benchmark benchmark/dompi_qa.jsonl \
    --output "avaliacao/resultados_baseline_teresina.json"

# Passo 3: pré-treino continuado (full-FT, idêntico ao Job 384 exceto pelo corpus)
# lr=2e-6 — mesmo do Job 384, para comparação direta de corpus
# ~9.100 docs de treino → ~14M tokens → ~13.000 blocos de 1024 tokens → ~810 steps
echo "--- Iniciando pré-treino (Teresina, lr=2e-6)..."
$PYTHON treino/pretreino_continuado.py \
    --model "$MODEL" \
    --train-data "$DATA_DIR/train_corpus.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --block-size 1024 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 2e-6 \
    --warmup-fraction 0.05

# Passo 4: avaliação pós-treino no held-out de Teresina
echo "--- Avaliando pós-treino (held-out Teresina)..."
EVAL_CKPT="$OUTPUT_DIR/final"
if [ ! -d "$EVAL_CKPT" ]; then
    EVAL_CKPT="$OUTPUT_DIR/best"
    echo "[aviso] final ausente, usando best"
fi
$PYTHON avaliacao/avaliar_modelo.py \
    --model "$(realpath $EVAL_CKPT)" \
    --held-out "$DATA_DIR/held_out.jsonl" \
    --benchmark benchmark/dompi_qa.jsonl \
    --output "$RESULTS_POSTREINO"

# Passo 5: inferências comparativas (baseline vs Teresina-treinado)
echo "--- Inferências comparativas..."
$PYTHON avaliacao/inferencia_multi.py \
    --models "baseline:$MODEL" "job389_teresina_lr2e6:$(realpath $EVAL_CKPT)" \
    --output "avaliacao/inferencias_job389_teresina.json" \
    --max-new-tokens 200

# Passo 6: relatório
echo "--- Relatório antes x depois (Teresina):"
$PYTHON avaliacao/comparar_resultados.py \
    --baseline "avaliacao/resultados_baseline_teresina.json" \
    --postreino "$RESULTS_POSTREINO"

echo ""
echo "=== Job 389 concluído: $(date) ==="
echo ""
echo "Interpretação da hipótese:"
echo "  Se CE pós-treino < CE baseline (Teresina) → corpus OCR é a causa raiz ✓"
echo "  Se CE pós-treino > CE baseline (Teresina) → problema é arquitetural/LR   ✗"
