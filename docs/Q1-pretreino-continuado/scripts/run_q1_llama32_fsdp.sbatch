#!/bin/bash
#SBATCH --job-name=q1-llama32-fsdp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:l4:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=trabalho-final/Q1-pretreino-continuado/resultados/log_llama32_fsdp_%j.txt
#SBATCH --error=trabalho-final/Q1-pretreino-continuado/resultados/log_llama32_fsdp_%j.txt

# Q1 — Pré-treino continuado CROSS-FAMÍLIA: Llama-3.2-3B, full-FT DISTRIBUÍDO (FSDP, 2xL4).
# Usa as DUAS GPUs do nó (capacidade total por tarefa). Corpus DOM-PI LIMPO.
# Modelo: pesos Llama-3.2-3B (Meta); baixados via mirror ungated (a org meta-llama
# está gated p/ este token). No relatório: "Llama-3.2-3B (família Meta)".
# Submeter da raiz do repo: ~/building-a-LLM  (entra na fila até o 538 liberar gpunode01)

set -e
REPO="$HOME/building-a-LLM"
cd "$REPO"
PYTHON="$REPO/.venv/bin/python3"
Q1="$REPO/trabalho-final/Q1-pretreino-continuado"
SCR="$Q1/scripts"
BENCH="$Q1/dompi_qa.jsonl"
RES="$Q1/resultados"
MODEL="unsloth/Llama-3.2-3B"     # pesos Meta, mirror ungated
TRAIN="$REPO/data/train_corpus_limpo.jsonl"
HELD_OUT="$REPO/data/held_out.jsonl"
OUTPUT_DIR="$REPO/treino/checkpoints_llama32_3b_dompi"
mkdir -p "$RES" "$REPO/treino"

echo "=== Q1 cross-família — Llama-3.2-3B full-FT FSDP (2xL4) ==="
echo "Node: $SLURMD_NODENAME | Data: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
$PYTHON -c "import accelerate,bitsandbytes,datasets;print('libs ok')"

# 1. Treino distribuído (2 processos = 2 GPUs)
echo "--- [1] torchrun FSDP (nproc_per_node=2)..."
$PYTHON -m torch.distributed.run --nproc_per_node=2 --master_port=29501 \
    "$SCR/pretreino_fsdp.py" \
    --model "$MODEL" \
    --train-data "$TRAIN" \
    --held-out "$HELD_OUT" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 1 --block-size 512 \
    --per-device-batch 2 --grad-accum 4 \
    --lr 2e-6 --warmup-fraction 0.05 \
    --eval-every 250 --patience 3 \
    --wrap-layer LlamaDecoderLayer

# 2. Avaliação antes×depois (single-GPU; usa 1 das 2 L4)
BEST="$(realpath "$OUTPUT_DIR/best")"
echo "--- [2] Avaliação baseline (Llama-3.2-3B) vs DAPT"
CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCR/avaliar_modelo.py" --model "$MODEL" \
    --held-out "$HELD_OUT" --benchmark "$BENCH" \
    --output "$RES/dominio_llama32_baseline.json" --max-samples 1000
CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCR/avaliar_modelo.py" --model "$BEST" \
    --held-out "$HELD_OUT" --benchmark "$BENCH" \
    --output "$RES/dominio_llama32_dapt.json" --max-samples 1000
CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCR/avaliar_geracao.py" --model "$MODEL" \
    --benchmark "$BENCH" --output "$RES/geracao_llama32_baseline.json"
CUDA_VISIBLE_DEVICES=0 $PYTHON "$SCR/avaliar_geracao.py" --model "$BEST" \
    --benchmark "$BENCH" --output "$RES/geracao_llama32_dapt.json"

echo "=================================================================="
echo "RESULTADO Llama-3.2-3B (antes -> depois):"
$PYTHON -c "import json;b=json.load(open('$RES/dominio_llama32_baseline.json'));d=json.load(open('$RES/dominio_llama32_dapt.json'));print(f'  PPL_dominio {b[\"perplexity\"]} -> {d[\"perplexity\"]} | CE {b[\"cross_entropy\"]} -> {d[\"cross_entropy\"]} | tokacc {b[\"token_accuracy\"]} -> {d[\"token_accuracy\"]}')"
echo "=== Concluído ==="
