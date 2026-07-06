#!/bin/bash
#SBATCH --job-name=q2q3-juiz
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 → TP=2 do juiz 14B
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/q2q3_juiz_%j.out

# LLM-as-judge (Qwen2.5-14B) sobre TODAS as gerações do benchmark (baseline +
# 6 modelos treinados, nos 2 tamanhos). Roda após run_q2q3_treino_aval.

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCR=trabalho-final/Q2-pos-treino-sft/scripts
RES=trabalho-final/Q2-pos-treino-sft/resultados

for sz in 1.5b 0.5b; do
  for tag in baseline_$sz full_$sz lora_$sz qlora_$sz; do
    GEN=$RES/bench_$tag.json
    if [ -f "$GEN" ]; then
      echo ">> Juiz: $tag"
      $PY_GEN $SCR/avaliar_juiz.py --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
          --gen-json "$GEN" --output "$RES/juiz_$tag.json"
    fi
  done
done

echo ">> OK — notas do juiz em $RES/juiz_*.json"
