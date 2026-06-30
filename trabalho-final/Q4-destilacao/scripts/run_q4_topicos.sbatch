#!/bin/bash
#SBATCH --job-name=q4-topicos
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 → tensor-parallel=2 p/ o professor 14B
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/q4_topicos_%j.out

# Q4 v2 — destilação SECCIONADA POR TÓPICO (DOM-PI × docentesDC), com CONTEXTO-OURO
# (passagem-fonte como contexto → zero "Não consta"). Gera dataset+benchmark factuais,
# destila 4 especialistas {dompi,docentes} × {0.5B,1.5B} (melhor receita = combined) e avalia
# cada um no benchmark do seu próprio tópico.
set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python     # vllm (professor 14B)
PY=.venv/bin/python               # transformers (destilação/avaliação)
export PATH="$HOME/building-a-LLM/.venv-q4gen/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

Q4=trabalho-final/Q4-destilacao
DADOS=$Q4/dados_v2
SCR=$Q4/scripts
MODELOS=$Q4/modelos
RES=$Q4/resultados
mkdir -p "$DADOS" "$RES"

# 1) Dataset de treino (contexto-ouro, sem RAG) — ambas as fontes, com top-50 logits
echo ">> [1/5] Gerando dataset de treino (contexto-ouro)..."
$PY_GEN $SCR/gerar_dataset_destilacao.py \
    --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 --max-model-len 8192 \
    --n-dompi 400 --n-docentes 400 --topk 50 --gold-context \
    --dompi-seeds data/held_out.jsonl \
    --out-dir "$DADOS"

# 2) Benchmark held-out (contexto-ouro, 50+50, sem abstenções)
echo ">> [2/5] Gerando benchmark held-out (contexto-ouro)..."
$PY_GEN $SCR/gerar_benchmark.py \
    --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 --max-model-len 8192 \
    --n-treino-dompi 400 --n-treino-docentes 400 --n-bench 50 --gold-context \
    --dompi-seeds data/held_out.jsonl \
    --out "$DADOS/benchmark_v2_100.jsonl"

# 3) Split por tópico (dataset+logits alinhados por id) e benchmark por tópico
echo ">> [3/5] Seccionando por tópico..."
$PY - "$DADOS" <<'PYEOF'
import json, sys, pathlib
D = pathlib.Path(sys.argv[1])
ds = [json.loads(l) for l in open(D/"dataset_G.jsonl")]
lg = {json.loads(l)["id"]: l for l in open(D/"logits_G.jsonl")}
for src, tag in [("DOM-PI","dompi"), ("docentesDC","docentes")]:
    n = 0
    with open(D/f"dataset_{tag}.jsonl","w") as fd, open(D/f"logits_{tag}.jsonl","w") as fl:
        for r in ds:
            if r["source"] == src:
                fd.write(json.dumps(r, ensure_ascii=False)+"\n"); fl.write(lg[r["id"]]); n += 1
    print(f"   treino {tag}: {n} exemplos")
bench = [json.loads(l) for l in open(D/"benchmark_v2_100.jsonl")]
for src, tag in [("DOM-PI","dompi"), ("docentesDC","docentes")]:
    with open(D/f"benchmark_{tag}.jsonl","w") as f:
        m = sum(f.write(json.dumps(r, ensure_ascii=False)+"\n") is not None for r in bench if r["source"]==src)
    print(f"   bench  {tag}: {m} perguntas")
PYEOF

# 4) Destila 4 especialistas (2 por vez, 1 por GPU) — melhor receita = combined
echo ">> [4/5] Destilando 4 especialistas por tópico..."
declare -A SIZE=( [0.5b]=Qwen/Qwen2.5-0.5B [1.5b]=Qwen/Qwen2.5-1.5B )
i=0
for tag in dompi docentes; do
  for sz in 0.5b 1.5b; do
    gpu=$(( i % 2 ))
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCR/destilar.py \
        --student "${SIZE[$sz]}" --braco B --method combined \
        --dataset "$DADOS/dataset_${tag}.jsonl" --logits "$DADOS/logits_${tag}.jsonl" \
        --out-dir "$MODELOS/esp_${tag}_${sz}" &
    i=$(( i + 1 ))
    [ $gpu -eq 1 ] && wait    # despacha em pares (2 GPUs)
  done
done
wait

# 5) Avalia cada especialista no benchmark do SEU tópico (+ bases)
echo ">> [5/5] Avaliando por tópico..."
for tag in dompi docentes; do
  $PY $SCR/avaliar_destilacao.py \
      --benchmark "$DADOS/benchmark_${tag}.jsonl" \
      --out "$RES/avaliacao_${tag}.json" \
      --model "base_05=Qwen/Qwen2.5-0.5B" --model "base_15=Qwen/Qwen2.5-1.5B" \
      --model "esp_${tag}_0.5b=$MODELOS/esp_${tag}_0.5b" \
      --model "esp_${tag}_1.5b=$MODELOS/esp_${tag}_1.5b"
done
echo "=== Q4 por tópico concluída → $RES/avaliacao_{dompi,docentes}.json ==="
