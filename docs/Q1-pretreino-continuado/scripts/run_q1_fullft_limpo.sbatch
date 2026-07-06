#!/bin/bash
#SBATCH --job-name=q1-fullft-limpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=trabalho-final/Q1-pretreino-continuado/resultados/log_fullft_limpo_%j.txt
#SBATCH --error=trabalho-final/Q1-pretreino-continuado/resultados/log_fullft_limpo_%j.txt

# Q1 — Eixo A: DAPT no corpus unificado APÓS LIMPEZA HEURÍSTICA DE OCR.
# Hipótese: sinal de treino mais limpo rende mais ganho por token (como no
# subcorpus curado de Teresina). Held-out FIXO (mesmo do -11,3% atual) para
# comparabilidade direta. Submeter da raiz do repo: ~/building-a-LLM
#
# Mesmos hiperparâmetros do full-FT unificado de referência:
#   epochs 1, block 512, batch 2, grad-accum 8, lr 2e-6, warmup 5%, patience 3.

set -e
REPO="$HOME/building-a-LLM"
cd "$REPO"
PYTHON="$REPO/.venv/bin/python3"
Q1="$REPO/trabalho-final/Q1-pretreino-continuado"
SCR="$Q1/scripts"
BENCH="$Q1/dompi_qa.jsonl"
RES="$Q1/resultados"
MODEL="Qwen/Qwen2.5-1.5B"
TRAIN_RAW="$REPO/data/train_corpus.jsonl"
TRAIN_LIMPO="$REPO/data/train_corpus_limpo.jsonl"
HELD_OUT="$REPO/data/held_out.jsonl"
OUTPUT_DIR="$REPO/treino/checkpoints_fullft_unif_limpo"
mkdir -p "$RES" "$REPO/treino"

echo "=== Q1 Eixo A — DAPT unificado-limpo ==="
echo "Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[aviso] nvidia-smi indisponível"

# 1. Limpeza de OCR (CPU, rápida) — só se ainda não existir
if [ ! -f "$TRAIN_LIMPO" ]; then
  echo "--- [1] Limpando OCR do corpus de treino..."
  $PYTHON "$SCR/limpar_corpus_ocr.py" \
      --input "$TRAIN_RAW" --output "$TRAIN_LIMPO" \
      --stats "$RES/limpeza_stats.json"
fi

# 2. DAPT full-FT no corpus limpo (held-out fixo)
echo "--- [2] Treino full-FT no corpus limpo..."
$PYTHON "$SCR/pretreino_fullft.py" \
    --model "$MODEL" \
    --train-data "$TRAIN_LIMPO" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 \
    --block-size 512 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 2e-6 \
    --warmup-fraction 0.05 \
    --gradient-checkpointing \
    --eval-every 250 \
    --patience 3

# 3. Avaliação do melhor checkpoint (mesmo held-out + benchmark + geração)
EVAL_CKPT="$OUTPUT_DIR/best"; [ -d "$EVAL_CKPT" ] || EVAL_CKPT="$OUTPUT_DIR/final"
EVAL_CKPT="$(realpath "$EVAL_CKPT")"
echo "--- [3] Avaliando $EVAL_CKPT"
$PYTHON "$SCR/avaliar_modelo.py" --model "$EVAL_CKPT" \
    --held-out "$HELD_OUT" --benchmark "$BENCH" \
    --output "$RES/dominio_unif_limpo.json" --max-samples 1000
$PYTHON "$SCR/avaliar_geracao.py" --model "$EVAL_CKPT" \
    --benchmark "$BENCH" --output "$RES/geracao_unif_limpo.json"

echo "=================================================================="
echo "RESULTADO unificado-limpo (comparar com baseline PPL_dominio e -11,3%):"
$PYTHON -c "import json;d=json.load(open('$RES/dominio_unif_limpo.json'));print(f'  PPL_dominio={d[\"perplexity\"]} CE={d[\"cross_entropy\"]} tokacc={d[\"token_accuracy\"]}')"
$PYTHON -c "import json;d=json.load(open('$RES/geracao_unif_limpo.json'));print(f'  geracao contains={d[\"gen_contains\"]} f1={d[\"gen_f1_mean\"]} EM={d[\"gen_exact_match\"]}')"
$PYTHON -c "import json;d=json.load(open('$RES/limpeza_stats.json'));print(f'  limpeza: docs {d[\"docs_in\"]}->{d[\"docs_out\"]} ({d[\"doc_keep_rate\"]*100:.0f}%), chars {d[\"char_keep_rate\"]*100:.0f}%')"
echo "=== Concluído ==="
