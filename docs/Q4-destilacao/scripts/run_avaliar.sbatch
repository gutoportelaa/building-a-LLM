#!/bin/bash
#SBATCH --job-name=q4-avaliar
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/q4_avaliar_%j.out

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python     # benchmark: vllm (professor 14B)
PY_EVAL=.venv/bin/python          # avaliação: transformers (alunos pequenos)
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DADOS=trabalho-final/Q4-destilacao/dados
MODELOS=trabalho-final/Q4-destilacao/modelos
SCR=trabalho-final/Q4-destilacao/scripts
BENCH=$DADOS/benchmark_destilacao_100.jsonl

# 1) Benchmark held-out (50 DOM-PI + 50 docentes), referência = professor 14B com RAG
if [ ! -f "$BENCH" ]; then
  echo ">> Gerando benchmark held-out..."
  $PY_GEN "$SCR/gerar_benchmark.py" \
      --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 --max-model-len 8192 \
      --n-treino-dompi 500 --n-treino-docentes 500 --n-bench 50 \
      --dompi-seeds data/held_out.jsonl --index-dir rag/index \
      --rag-scripts trabalho-final/Q5-rag/scripts --out "$BENCH"
fi

# 2) Monta a lista de modelos: 2 bases + 12 alunos destilados
ARGS=(--benchmark "$BENCH" --out trabalho-final/Q4-destilacao/resultados/avaliacao.json)
ARGS+=(--model "base_05=Qwen/Qwen2.5-0.5B" --model "base_15=Qwen/Qwen2.5-1.5B")
for size in 0.5b 1.5b; do
  for br in A B; do
    for m in ce kl combined; do
      d="$MODELOS/aluno_qwen2.5-${size}_${br}_${m}"
      [ -d "$d" ] && ARGS+=(--model "d_${size}_${br}_${m}=$d")
    done
  done
done

echo ">> Avaliando $(( (${#ARGS[@]}-4)/2 )) modelos no benchmark..."
$PY_EVAL "$SCR/avaliar_destilacao.py" "${ARGS[@]}"
echo "=== avaliação concluída → trabalho-final/Q4-destilacao/resultados/avaliacao.json ==="
