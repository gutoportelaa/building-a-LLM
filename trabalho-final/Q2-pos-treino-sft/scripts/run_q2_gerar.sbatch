#!/bin/bash
#SBATCH --job-name=q2-gerar-pares
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 → tensor-parallel=2 do professor 14B
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/q2_gerar_%j.out

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python                    # vllm (professor 14B), idêntico à Q4
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCR=trabalho-final/Q2-pos-treino-sft/scripts
DADOS=trabalho-final/Q2-pos-treino-sft/dados

# 1) PILOTO (~100 pares, sem juiz) — valida prompt/qualidade rapidamente.
echo ">> PILOTO (100 pares)"
$PY_GEN $SCR/gerar_pares_docentes.py --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
    --n-pairs 100 --pilot --out-dir $DADOS/piloto

# 2) LOTE CHEIO (>=1000 limpos; gera ~2400, juiz filtra score>=3, corta em 1500).
echo ">> LOTE CHEIO (alvo 1500 limpos)"
$PY_GEN $SCR/gerar_pares_docentes.py --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
    --n-pairs 1500 --oversample 1.6 --min-score 3 \
    --out-dir $DADOS

echo ">> OK — pares em $DADOS/ (pares_train.jsonl / pares_heldout.jsonl)"
