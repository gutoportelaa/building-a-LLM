#!/bin/bash
#SBATCH --job-name=q2q3-juiz-1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                 # 14B-AWQ em 1 L4 → não espera o nó de 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/q2q3_juiz_1gpu_%j.out

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCR=trabalho-final/Q2-pos-treino-sft/scripts
RES=trabalho-final/Q2-pos-treino-sft/resultados
TEACHER=Qwen/Qwen2.5-14B-Instruct-AWQ

for sz in 1.5b 0.5b; do
  for tag in baseline_$sz full_$sz lora_$sz qlora_$sz; do
    GEN=$RES/bench_$tag.json
    if [ -f "$GEN" ]; then
      echo ">> Juiz: $tag"
      $PY_GEN $SCR/avaliar_juiz.py --teacher $TEACHER --tp 1 --quantization awq \
          --gen-json "$GEN" --output "$RES/juiz_$tag.json"
    fi
  done
done

echo ">> OK — notas do juiz em $RES/juiz_*.json"
# Consolidação final (tabelas + figuras)
.venv/bin/python $SCR/consolidar_q2q3.py
