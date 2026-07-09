#!/bin/bash
#SBATCH --job-name=q4-aluno-xfam
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 — um aluno por GPU
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/q4_aluno_xfam_%j.out

# Q4 — desdobramento (b) da §4.10b do relatório: ALUNOS DE OUTRA FAMÍLIA (Llama-3.2),
# black-box (CE no texto do professor Qwen-14B, braço B). Fecha a célula vazia do
# quadro 2×2 white-box × black-box:
#   (black-box aluno Qwen 1.5B: KR 0,647)  −  (black-box aluno Llama)  =  custo de trocar
#   a família do aluno, com professor/dataset/sinal idênticos.
#
# Não roda o professor: reusa dados/dataset_B.jsonl (texto), re-tokenizado por aluno.
# KL é impossível entre tokenizadores distintos — é exatamente o que o experimento isola.
set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY=.venv/bin/python               # transformers (conversão/treino/avaliação)

Q4=trabalho-final/Q4-destilacao
DADOS=$Q4/dados
SCR=$Q4/scripts
MODELOS=$Q4/modelos
RES=$Q4/resultados

declare -A STUDENT=( [llama1b]=unsloth/Llama-3.2-1B [llama3b]=unsloth/Llama-3.2-3B )

# 1) Re-tokeniza o dataset B no espaço de cada aluno (só CPU, rápido)
echo ">> [1/3] Convertendo dataset_B para o espaço dos alunos Llama..."
for tag in llama1b llama3b; do
  $PY $SCR/preparar_dataset_aluno_xfam.py \
      --student "${STUDENT[$tag]}" \
      --dataset "$DADOS/dataset_B.jsonl" \
      --out "$DADOS/dataset_B_${tag}.jsonl"
done

# 2) Destila (CE / black-box), um aluno por GPU — mesma receita dos alunos Qwen
#    (3 épocas, lr 1e-5, grad-accum 16, prompt mascarado, labels sem pré-shift)
echo ">> [2/3] Treinando alunos cross-família (CE, braço B)..."
CUDA_VISIBLE_DEVICES=0 $PY $SCR/destilar.py \
    --student "${STUDENT[llama1b]}" --braco B --method ce \
    --dataset "$DADOS/dataset_B_llama1b.jsonl" \
    --out-dir "$MODELOS/xfam_llama1b_B_ce" &
CUDA_VISIBLE_DEVICES=1 $PY $SCR/destilar.py \
    --student "${STUDENT[llama3b]}" --braco B --method ce \
    --dataset "$DADOS/dataset_B_llama3b.jsonl" \
    --out-dir "$MODELOS/xfam_llama3b_B_ce" &
wait

# 3) Avalia bases e alunos no MESMO benchmark v1 de 100 perguntas dos alunos Qwen
#    (comparável linha a linha com avaliacao.json)
echo ">> [3/3] Avaliando alunos cross-família no benchmark v1..."
$PY $SCR/avaliar_destilacao.py \
    --benchmark "$DADOS/benchmark_destilacao_100.jsonl" \
    --out "$RES/avaliacao_alunos_xfam.json" \
    --model "base_llama1b=unsloth/Llama-3.2-1B" \
    --model "base_llama3b=unsloth/Llama-3.2-3B" \
    --model "xfam_llama1b_B_ce=$MODELOS/xfam_llama1b_B_ce" \
    --model "xfam_llama3b_B_ce=$MODELOS/xfam_llama3b_B_ce"
echo "=== q4-aluno-xfam concluído → $RES/avaliacao_alunos_xfam.json ==="
