#!/bin/bash
#SBATCH --job-name=q4-destilar
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 → 2 destilações em paralelo (1 aluno por GPU)
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/q4_destilar_%j.out

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_TRAIN=.venv/bin/python         # ambiente de treino (torch + transformers)
DADOS=trabalho-final/Q4-destilacao/dados
MODELOS=trabalho-final/Q4-destilacao/modelos
SCRIPT=trabalho-final/Q4-destilacao/scripts/destilar.py

# Matriz: 2 alunos × 3 métodos × 2 braços = 12 alunos destilados.
STUDENTS=("Qwen/Qwen2.5-0.5B" "Qwen/Qwen2.5-1.5B")
METHODS=("ce" "kl" "combined")
BRACOS=("A" "B")

# Distribui os jobs entre as 2 GPUs: GPU livre recebe o próximo da fila.
gpu=0
for st in "${STUDENTS[@]}"; do
  tag_st=$(basename "$st" | tr '[:upper:]' '[:lower:]')
  for br in "${BRACOS[@]}"; do
    for m in "${METHODS[@]}"; do
      logits_arg=""
      [ "$m" != "ce" ] && logits_arg="--logits $DADOS/logits_${br}.jsonl"
      out="$MODELOS/aluno_${tag_st}_${br}_${m}"
      echo ">> [GPU $gpu] $out"
      CUDA_VISIBLE_DEVICES=$gpu $PY_TRAIN "$SCRIPT" \
          --student "$st" --braco "$br" --method "$m" \
          --dataset "$DADOS/dataset_${br}.jsonl" $logits_arg \
          --out-dir "$out" &
      gpu=$(( (gpu + 1) % 2 ))
      [ "$gpu" -eq 0 ] && wait    # a cada par lançado, espera fechar antes do próximo
    done
  done
done
wait
echo "Todas as 12 destilações concluídas."
