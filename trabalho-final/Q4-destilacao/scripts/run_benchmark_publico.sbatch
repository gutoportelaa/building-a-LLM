#!/bin/bash
#SBATCH --job-name=q4-bench-pub
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/q4_bench_pub_%j.out

# Retenção ancorada da Q4 num benchmark PÚBLICO (ENEM PT-BR).
# Mede professor 14B + alunos por log-verossimilhança da alternativa (sem geração)
# e calcula retenção = acc_aluno / acc_professor (comparável ao "97% do BERT" do DistilBERT).
# Professor 14B carregado em 8-bit p/ caber em 1 L4 (24GB) — quase lossless p/ inferência.
set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY=.venv/bin/python
SCR=trabalho-final/Q4-destilacao/scripts
MODELOS=trabalho-final/Q4-destilacao/modelos
OUT=trabalho-final/Q4-destilacao/resultados/avaliacao_benchmark_publico.json

# professor (teto) + 2 bases + os melhores destilados de cada tamanho
ARGS=(--dataset eduagarcia/enem_challenge --split train --limit 200 --teacher prof --out "$OUT")
ARGS+=(--model "prof=Qwen/Qwen2.5-14B-Instruct")
ARGS+=(--model "base_05=Qwen/Qwen2.5-0.5B" --model "base_15=Qwen/Qwen2.5-1.5B")
for spec in "0.5b_B_combined" "1.5b_B_combined" "1.5b_B_kl"; do
  d="$MODELOS/aluno_qwen2.5-${spec}"
  [ -d "$d" ] && ARGS+=(--model "d_${spec}=$d")
done

echo ">> Avaliando retenção em benchmark público (ENEM)..."
$PY "$SCR/avaliar_benchmark_publico.py" "${ARGS[@]}"
echo "=== concluído → $OUT ==="
