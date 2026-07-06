#!/bin/bash
#SBATCH --job-name=q4-aval-bxf
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/q4_aval_bxf_%j.out

# Avalia os 2 alunos cross-família (Plano B, zephyr→Qwen) no MESMO benchmark de 100Q,
# para o contraste white-box (mesma família) × black-box (cross-família).

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_EVAL=.venv/bin/python
DADOS=trabalho-final/Q4-destilacao/dados
MOD=trabalho-final/Q4-destilacao/modelos
SCR=trabalho-final/Q4-destilacao/scripts

$PY_EVAL "$SCR/avaliar_destilacao.py" \
    --benchmark "$DADOS/benchmark_destilacao_100.jsonl" \
    --out trabalho-final/Q4-destilacao/resultados/avaliacao_bxf.json \
    --model "base_15=Qwen/Qwen2.5-1.5B" \
    --model "bxf_0.5b_ce=$MOD/aluno_qwen2.5-0.5b_Bxf_ce" \
    --model "bxf_1.5b_ce=$MOD/aluno_qwen2.5-1.5b_Bxf_ce"

echo "=== avaliação cross-família concluída → resultados/avaliacao_bxf.json ==="
