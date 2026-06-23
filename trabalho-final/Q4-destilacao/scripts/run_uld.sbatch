#!/bin/bash
#SBATCH --job-name=q4-uld
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/q4_uld_%j.out

# ULD cross-tokenizer (fronteira): logit KD entre famílias (zephyr→Qwen) via distribuições de probabilidade
# ORDENADAS (sem vocabulário comum). Geração com top-k logprobs do zephyr → destilação uldcomb → avaliação.

set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python
PY_TRAIN=.venv/bin/python
PY_EVAL=.venv/bin/python
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

DADOS=trabalho-final/Q4-destilacao/dados
MOD=trabalho-final/Q4-destilacao/modelos
SCR=trabalho-final/Q4-destilacao/scripts

# 1) Geração zephyr + TOP-50 LOGPROBS (espaço-zephyr) — reusa questões+contexto B
if [ ! -f "$DADOS/logits_Bxf.jsonl" ]; then
  $PY_GEN "$SCR/gerar_dataset_crossfamilia.py" \
      --teacher HuggingFaceH4/zephyr-7b-beta --tp 2 --max-model-len 8192 --topk 50 \
      --questoes "$DADOS/questoes.jsonl" --contexto "$DADOS/contexto_B.jsonl" \
      --out "$DADOS/dataset_Bxf_uld.jsonl" --logits-out "$DADOS/logits_Bxf.jsonl"
fi

# 2) Destilação ULD (uldcomb = α·CE + (1−α)·ULD) — 0.5B e 1.5B
for st in 0.5B 1.5B; do
  tag=$(echo "$st" | tr '[:upper:]' '[:lower:]')
  echo ">> destilando aluno $st (uldcomb cross-tokenizer)"
  $PY_TRAIN "$SCR/destilar.py" \
      --student "Qwen/Qwen2.5-$st" --braco B --method uldcomb \
      --dataset "$DADOS/dataset_Bxf_uld.jsonl" --logits "$DADOS/logits_Bxf.jsonl" \
      --out-dir "$MOD/aluno_uld_${tag}"
done

# 3) Avaliação no mesmo benchmark de 100Q
ARGS=(--benchmark "$DADOS/benchmark_destilacao_100.jsonl" \
      --out trabalho-final/Q4-destilacao/resultados/avaliacao_uld.json \
      --model "base_15=Qwen/Qwen2.5-1.5B")
[ -d "$MOD/aluno_uld_0.5b" ] && ARGS+=(--model "uld_0.5b=$MOD/aluno_uld_0.5b")
[ -d "$MOD/aluno_uld_1.5b" ] && ARGS+=(--model "uld_1.5b=$MOD/aluno_uld_1.5b")
$PY_EVAL "$SCR/avaliar_destilacao.py" "${ARGS[@]}"

echo "=== ULD concluído → resultados/avaliacao_uld.json ==="
