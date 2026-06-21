#!/bin/bash
#SBATCH --job-name=q4-gerar-dataset
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 → tensor-parallel=2 para o professor 14B
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/q4_gerar_%j.out

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python     # vllm (professor) + sentence-transformers (índice/recuperação)
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"   # ninja no PATH p/ JIT do flashinfer (nvcc/gcc já no sistema)

# Pré-requisito do braço B: índice RAG da Q5 (regenerável, ~30 min).
# Rode UMA vez se rag/index/embeddings.npy não existir:
if [ ! -f rag/index/embeddings.npy ]; then
  echo ">> Reconstruindo índice RAG (braço B)..."
  $PY_GEN trabalho-final/Q5-rag/scripts/build_index.py \
      --corpus data/train_corpus.jsonl data/held_out.jsonl \
      --out-dir rag/index
fi

$PY_GEN trabalho-final/Q4-destilacao/scripts/gerar_dataset_destilacao.py \
    --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
    --n-dompi 500 --n-docentes 500 --topk 50 --bracos A B \
    --dompi-seeds data/held_out.jsonl \
    --index-dir rag/index \
    --rag-scripts trabalho-final/Q5-rag/scripts \
    --out-dir trabalho-final/Q4-destilacao/dados
