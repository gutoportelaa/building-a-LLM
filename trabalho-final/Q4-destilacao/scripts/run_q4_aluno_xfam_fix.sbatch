#!/bin/bash
#SBATCH --job-name=q4-xfam-fix
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/q4_xfam_fix_%j.out

# Retomada do q4-aluno-xfam: o 1B treinou e foi salvo; o 3B estourou a VRAM (AdamW fp32).
# Correção: --optim adamw8bit (bitsandbytes) — a mesma receita que coube o Llama-3.2-3B
# full-FT na L4 durante a Q1. O dataset convertido (dataset_B_llama3b.jsonl) já está em disco.
set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY=.venv/bin/python
Q4=trabalho-final/Q4-destilacao
DADOS=$Q4/dados
SCR=$Q4/scripts
MODELOS=$Q4/modelos
RES=$Q4/resultados

# 1) Treina só o 3B, agora com otimizador 8-bit
echo ">> [1/2] Treinando aluno Llama-3.2-3B (CE, braço B, AdamW-8bit)..."
$PY $SCR/destilar.py \
    --student unsloth/Llama-3.2-3B --braco B --method ce --optim adamw8bit \
    --dataset "$DADOS/dataset_B_llama3b.jsonl" \
    --out-dir "$MODELOS/xfam_llama3b_B_ce"

# 2) Avaliação completa dos 4 modelos no benchmark v1 (comparável a avaliacao.json)
echo ">> [2/2] Avaliando bases e alunos cross-família..."
$PY $SCR/avaliar_destilacao.py \
    --benchmark "$DADOS/benchmark_destilacao_100.jsonl" \
    --out "$RES/avaliacao_alunos_xfam.json" \
    --model "base_llama1b=unsloth/Llama-3.2-1B" \
    --model "base_llama3b=unsloth/Llama-3.2-3B" \
    --model "xfam_llama1b_B_ce=$MODELOS/xfam_llama1b_B_ce" \
    --model "xfam_llama3b_B_ce=$MODELOS/xfam_llama3b_B_ce"
echo "=== q4-xfam-fix concluído → $RES/avaliacao_alunos_xfam.json ==="
