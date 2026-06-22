#!/bin/bash
#SBATCH --job-name=q4-aval-dapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/q4_aval_dapt_%j.out

# Avalia os alunos DAPT-then-distill + o DAPT cru + base no benchmark de 100Q.
# Comparação: o priming de domínio (DAPT) ajuda a destilação? (vs alunos base d_1.5b_B_combined / bxf_1.5b_ce)
set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_EVAL=.venv/bin/python
DADOS=trabalho-final/Q4-destilacao/dados
MOD=trabalho-final/Q4-destilacao/modelos
SCR=trabalho-final/Q4-destilacao/scripts
DAPT="$HOME/building-a-LLM/treino/checkpoints_fullft_unificado_v2/best"

ARGS=(--benchmark "$DADOS/benchmark_destilacao_100.jsonl" \
      --out trabalho-final/Q4-destilacao/resultados/avaliacao_dapt.json \
      --model "base_15=Qwen/Qwen2.5-1.5B" \
      --model "dapt_raw=$DAPT")
[ -d "$MOD/aluno_dapt_B_combined" ] && ARGS+=(--model "dapt_B_combined=$MOD/aluno_dapt_B_combined")
[ -d "$MOD/aluno_dapt_Bxf_ce" ]     && ARGS+=(--model "dapt_Bxf_ce=$MOD/aluno_dapt_Bxf_ce")

$PY_EVAL "$SCR/avaliar_destilacao.py" "${ARGS[@]}"
echo "=== avaliação DAPT-then-distill concluída → resultados/avaliacao_dapt.json ==="
