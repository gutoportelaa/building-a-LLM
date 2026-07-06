#!/bin/bash
#SBATCH --job-name=q4-aval-fut
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/q4_aval_fut_%j.out

# Avaliação do Plano A (Copa 2026): benchmark held-out de fatos + avalia os 4 alunos de raciocínio.
set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python
PY_EVAL=.venv/bin/python
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

FUT=trabalho-final/Q4-destilacao/futebol
SCR=trabalho-final/Q4-destilacao/scripts
MOD=trabalho-final/Q4-destilacao/modelos
BENCH=$FUT/benchmark_futebol.jsonl

# 1) Benchmark held-out de FATOS estáveis (seeds [200:] do corpus, disjuntos do treino).
#    Referência = Qwen2.5-14B-Instruct + RAG (concisa; key_recall é a métrica robusta vs respostas com <think>).
if [ ! -f "$BENCH" ]; then
  $PY_GEN "$SCR/gerar_benchmark.py" \
      --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 --max-model-len 8192 --max-new-tokens 256 \
      --dompi-seeds "$FUT/corpus_futebol.jsonl" --index-dir "$FUT/index" \
      --rag-scripts trabalho-final/Q5-rag/scripts \
      --n-treino-dompi 200 --n-bench 41 --min-seed-chars 40 --skip-docentes \
      --out "$BENCH"
fi

# 2) Avalia base + 4 alunos de raciocínio (max-new-tokens maior p/ caber <think>+resposta)
ARGS=(--benchmark "$BENCH" --max-new-tokens 512 \
      --out trabalho-final/Q4-destilacao/resultados/avaliacao_futebol.json \
      --model "base_15=Qwen/Qwen2.5-1.5B")
for st in 0.5b 1.5b; do
  for br in A B; do
    d="$MOD/aluno_qwen2.5-${st}_fut_${br}_combined"
    [ -d "$d" ] && ARGS+=(--model "fut_${st}_${br}=$d")
  done
done
$PY_EVAL "$SCR/avaliar_destilacao.py" "${ARGS[@]}"
echo "=== avaliação futebol concluída → resultados/avaliacao_futebol.json ==="
