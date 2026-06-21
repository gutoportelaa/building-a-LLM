#!/bin/bash
#SBATCH --job-name=q4-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 → TP=2 do professor 14B
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=logs/q4_smoke_%j.out

# Dry-run de fumaça: valida vLLM(TP=2)→top-k logprobs→dataset→destilação (CE/KL/combined)
# com pouquíssimos exemplos. Só braço A (não precisa do índice RAG). Também pré-aquece o
# cache de modelos do HF para a rodada cheia. Se isto passar, não há erro crasso no caminho.

set -euo pipefail
cd "$HOME/building-a-LLM"
source .venv/bin/activate
mkdir -p logs

DS=trabalho-final/Q4-destilacao/dados_smoke
MD=trabalho-final/Q4-destilacao/modelos_smoke
SCR=trabalho-final/Q4-destilacao/scripts

echo "=== [1/2] Geração mínima (braço A, 4+4 perguntas, top-50) ==="
python "$SCR/gerar_dataset_destilacao.py" \
    --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 \
    --n-dompi 4 --n-docentes 4 --topk 50 --bracos A \
    --dompi-seeds data/held_out.jsonl \
    --out-dir "$DS"

echo "=== [2/2] Destilação mínima (aluno 0.5B, 3 métodos, 1 época) ==="
for m in ce kl combined; do
  logits_arg=""
  [ "$m" != "ce" ] && logits_arg="--logits $DS/logits_A.jsonl"
  echo ">> método $m"
  python "$SCR/destilar.py" \
      --student Qwen/Qwen2.5-0.5B --braco A --method "$m" \
      --dataset "$DS/dataset_A.jsonl" $logits_arg \
      --out-dir "$MD/aluno_0.5b_A_$m" \
      --epochs 1 --grad-accum 2
done

echo "=== SMOKE OK — caminho completo validado. Pode submeter a rodada cheia. ==="
