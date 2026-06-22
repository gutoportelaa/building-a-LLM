#!/bin/bash
#SBATCH --job-name=q4-dapt-distill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/q4_dapt_distill_%j.out

# DAPT-then-distill: destila para o aluno JÁ adaptado ao domínio (Q1 Full FT v2), em vez do base pristino.
# Testa se o priming de domínio (DAPT) ajuda a destilação. gpu:1 sem fixar nó → pode usar o gpunode02 ocioso.

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_TRAIN=.venv/bin/python
DADOS=trabalho-final/Q4-destilacao/dados
MOD=trabalho-final/Q4-destilacao/modelos
SCR=trabalho-final/Q4-destilacao/scripts
DAPT="$HOME/building-a-LLM/treino/checkpoints_fullft_unificado_v2/best"   # Q1 DAPT 1.5B (Full FT v2)

# 1) white-box (combined) — reusa dataset_B + logits_B da Q4
echo ">> DAPT-then-distill white-box (combined)"
$PY_TRAIN "$SCR/destilar.py" --student "$DAPT" --braco B --method combined \
    --dataset "$DADOS/dataset_B.jsonl" --logits "$DADOS/logits_B.jsonl" \
    --out-dir "$MOD/aluno_dapt_B_combined"

# 2) cross-família (black-box, ce) — reusa dataset_Bxf (zephyr)
echo ">> DAPT-then-distill cross-família (ce)"
$PY_TRAIN "$SCR/destilar.py" --student "$DAPT" --braco B --method ce \
    --dataset "$DADOS/dataset_Bxf.jsonl" \
    --out-dir "$MOD/aluno_dapt_Bxf_ce"

echo "=== DAPT-then-distill concluído: aluno_dapt_B_combined + aluno_dapt_Bxf_ce ==="
