#!/bin/bash
#SBATCH --job-name=q2q3-treino-aval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/q2q3_treino_%j.out

# Treina os 3 métodos (full=Q2, lora/qlora=Q3) nos 2 modelos (1.5B + 0.5B) e
# avalia ANTES (base -Instruct) e DEPOIS, nos 2 modos (held-out intrínseca +
# benchmark geração). Tudo no .venv (transformers). Comparação controlada:
# os pares e o benchmark são idênticos entre todos os braços.

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY=.venv/bin/python
SCR=trabalho-final/Q2-pos-treino-sft/scripts
DADOS=trabalho-final/Q2-pos-treino-sft/dados
BENCH=trabalho-final/Q2-pos-treino-sft/benchmark/dc_bench.jsonl
MOD=trabalho-final/Q2-pos-treino-sft/modelos
RES=trabalho-final/Q2-pos-treino-sft/resultados
mkdir -p "$MOD" "$RES"

TRAIN=$DADOS/pares_train.jsonl
HELD=$DADOS/pares_heldout.jsonl

declare -A MODELS=( ["1.5b"]="Qwen/Qwen2.5-1.5B-Instruct" ["0.5b"]="Qwen/Qwen2.5-0.5B-Instruct" )

avaliar () {  # $1=model_path  $2=tag
  $PY $SCR/avaliar_sft.py --model "$1" --data "$HELD"  --modo intrinseca --output "$RES/heldout_$2.json"
  $PY $SCR/avaliar_sft.py --model "$1" --data "$BENCH" --modo geracao    --output "$RES/bench_$2.json" --max-new-tokens 256
}

for sz in 1.5b 0.5b; do
  BASE=${MODELS[$sz]}

  echo "=================== BASELINE (antes do SFT) — $sz ==================="
  avaliar "$BASE" "baseline_$sz"

  for m in full lora qlora; do
    echo "=================== TREINO $m — $sz ==================="
    LR=$([ "$m" = "full" ] && echo 1e-5 || echo 2e-4)
    $PY $SCR/sft_docentes.py --method $m --model "$BASE" \
        --train "$TRAIN" --out-dir "$MOD/sft_${m}_${sz}" \
        --epochs 3 --grad-accum 16 --lr $LR --max-len 1024
    echo "=================== AVALIAÇÃO $m — $sz ==================="
    avaliar "$MOD/sft_${m}_${sz}" "${m}_${sz}"
  done
done

echo ">> OK — métricas em $RES/ (heldout_*.json, bench_*.json) e modelos em $MOD/"
