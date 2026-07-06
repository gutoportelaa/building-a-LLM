#!/bin/bash
#SBATCH --job-name=q4-futebol
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/q4_futebol_%j.out

# Plano A — especialização temática Copa 2026 por DESTILAÇÃO DE RACIOCÍNIO (white-box logit KD).
# Professor: DeepSeek-R1-Distill-Qwen-14B (modelo de raciocínio sobre Qwen2.5-14B → MESMA família dos alunos →
# habilita logit KD). Reusa o pipeline da Q4 com o corpus/índice de futebol e comprimentos de raciocínio.

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python
PY_TRAIN=.venv/bin/python
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

TEACHER="${TEACHER:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
FUT=trabalho-final/Q4-destilacao/futebol
SCR=trabalho-final/Q4-destilacao/scripts
MOD=trabalho-final/Q4-destilacao/modelos

# 0) corpus + índice de futebol (se faltar no cluster)
[ -f "$FUT/corpus_futebol.jsonl" ] || $PY_GEN "$SCR/coletar_corpus_futebol.py" --out "$FUT/corpus_futebol.jsonl"
if [ ! -f "$FUT/index/embeddings.npy" ]; then
  $PY_GEN trabalho-final/Q5-rag/scripts/build_index.py --corpus "$FUT/corpus_futebol.jsonl" --out-dir "$FUT/index"
fi

# 1) geração de RACIOCÍNIO (professor R1 + RAG futebol), white-box (top-50 logits). Braços A (zerada) × B (RAG).
echo ">> Professor de raciocínio: $TEACHER"
$PY_GEN "$SCR/gerar_dataset_destilacao.py" \
    --teacher "$TEACHER" --tp 2 \
    --max-model-len 8192 --max-new-tokens 1536 --max-context-chars 6000 \
    --n-dompi 200 --n-docentes 0 --min-seed-chars 40 --topk 50 --bracos A B \
    --dompi-seeds "$FUT/corpus_futebol.jsonl" \
    --index-dir "$FUT/index" --rag-scripts trabalho-final/Q5-rag/scripts \
    --out-dir "$FUT/dados"

# 2) destilação white-box (combined) — 0.5B/1.5B × braços A/B (max-len alto p/ caber o <think>)
for st in 0.5B 1.5B; do
  tag=$(echo "$st" | tr '[:upper:]' '[:lower:]')
  for br in A B; do
    echo ">> destilando aluno $st braço $br (futebol/raciocínio)"
    $PY_TRAIN "$SCR/destilar.py" \
        --student "Qwen/Qwen2.5-$st" --braco "$br" --method combined --max-len 4096 \
        --dataset "$FUT/dados/dataset_${br}.jsonl" --logits "$FUT/dados/logits_${br}.jsonl" \
        --out-dir "$MOD/aluno_qwen2.5-${tag}_fut_${br}_combined"
  done
done
echo "=== Plano A (futebol) concluído: dataset + 4 alunos de raciocínio ==="
