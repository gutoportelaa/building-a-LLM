#!/bin/bash
#SBATCH --job-name=q4-crossfam
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/q4_crossfam_%j.out

# Plano B — cross-família black-box (sequence-KD).
# TEACHER pode ser sobrescrito: sbatch --export=ALL,TEACHER=mistralai/Mistral-7B-Instruct-v0.3 run_crossfamilia.sbatch

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python
PY_TRAIN=.venv/bin/python
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

TEACHER="${TEACHER:-google/gemma-2-9b-it}"
DADOS=trabalho-final/Q4-destilacao/dados
MOD=trabalho-final/Q4-destilacao/modelos
SCR=trabalho-final/Q4-destilacao/scripts

echo ">> Professor cross-família: $TEACHER"

# 1) Geração black-box (texto, sem logits) — reusa questões + contexto B da Q4
$PY_GEN "$SCR/gerar_dataset_crossfamilia.py" \
    --teacher "$TEACHER" --tp 2 --max-model-len 8192 \
    --questoes "$DADOS/questoes.jsonl" --contexto "$DADOS/contexto_B.jsonl" \
    --out "$DADOS/dataset_Bxf.jsonl"

# 2) Destilação SFT-texto (ce) para 0.5B e 1.5B
for st in 0.5B 1.5B; do
  tag=$(echo "$st" | tr '[:upper:]' '[:lower:]')
  echo ">> SFT cross-família aluno $st"
  $PY_TRAIN "$SCR/destilar.py" \
      --student "Qwen/Qwen2.5-$st" --braco B --method ce \
      --dataset "$DADOS/dataset_Bxf.jsonl" \
      --out-dir "$MOD/aluno_qwen2.5-${tag}_Bxf_ce"
done

echo "=== Plano B concluído: dataset_Bxf + 2 alunos cross-família (avaliar depois com run_avaliar) ==="
