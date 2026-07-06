#!/bin/bash
#SBATCH --job-name=q2-gerar-1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # 14B-AWQ cabe em 1 L4 → roda no nó ocioso (sem esperar o nó de 2 GPUs)
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/q2_gerar_1gpu_%j.out

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCR=trabalho-final/Q2-pos-treino-sft/scripts
DADOS=trabalho-final/Q2-pos-treino-sft/dados
TEACHER=Qwen/Qwen2.5-14B-Instruct-AWQ

# 1) PILOTO (~100 pares, sem juiz) — valida serving AWQ + prompt rapidamente.
echo ">> PILOTO (100 pares, 14B-AWQ, TP=1)"
$PY_GEN $SCR/gerar_pares_docentes.py --teacher $TEACHER --tp 1 --quantization awq \
    --max-model-len 4096 --n-pairs 100 --pilot --out-dir $DADOS/piloto

# 2) LOTE CHEIO (>=1000 limpos; juiz filtra score>=3; corta em 1500).
echo ">> LOTE CHEIO (alvo 1500 limpos)"
$PY_GEN $SCR/gerar_pares_docentes.py --teacher $TEACHER --tp 1 --quantization awq \
    --max-model-len 4096 --n-pairs 1500 --oversample 1.6 --min-score 3 \
    --out-dir $DADOS

echo ">> OK — pares em $DADOS/"
