#!/bin/bash
#SBATCH --job-name=q4-bench200
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode01        # 2× L4 → tensor-parallel=2 p/ o professor 14B
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/q4_bench200_%j.out

# Q4 — desdobramento (a) da §4.10b do relatório: benchmark de 100 HELD-OUT POR TÓPICO
# (200 no total) com contexto-ouro, para reduzir o erro-padrão do key_recall de ~±0,07
# para ~±0,045 e tirar os deltas menores (+10% DOM-PI) da zona de ruído.
#
# Lição da §4.8b aplicada POR CONSTRUÇÃO: além das fatias held-out determinísticas do
# gerador (disjuntas do treino por seed), um pós-filtro remove perguntas quase-duplicadas
# das 800 questões de treino v2 (Jaccard de palavras > 0,7) ANTES de fechar as 100/tópico.
#
# Não retreina nada: gera o benchmark (professor 14B) e reavalia os 6 modelos existentes.
set -euo pipefail
cd "$HOME/building-a-LLM"
mkdir -p logs

PY_GEN=.venv-q4gen/bin/python     # vllm (professor 14B)
PY=.venv/bin/python               # transformers (avaliação)
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

Q4=trabalho-final/Q4-destilacao
DADOS=$Q4/dados_v2
SCR=$Q4/scripts
MODELOS=$Q4/modelos
RES=$Q4/resultados

# 1) Benchmark held-out ampliado (contexto-ouro). n-bench 110/tópico para sobrar margem
#    ao pós-filtro de quase-duplicatas; fatias começam após as 400 sementes de treino v2.
echo ">> [1/3] Gerando benchmark ampliado (alvo: 110/tópico antes do filtro)..."
$PY_GEN $SCR/gerar_benchmark.py \
    --teacher Qwen/Qwen2.5-14B-Instruct --tp 2 --max-model-len 8192 \
    --n-treino-dompi 400 --n-treino-docentes 400 --n-bench 110 --gold-context \
    --dompi-seeds data/held_out.jsonl \
    --out "$DADOS/benchmark_v2_raw220.jsonl"

# 2) Pós-filtro de quase-duplicatas vs treino + corte em 100/tópico + split por tópico
echo ">> [2/3] Filtrando quase-duplicatas do treino e fechando 100/tópico..."
$PY - "$DADOS" <<'PYEOF'
import json, re, sys, pathlib
D = pathlib.Path(sys.argv[1])
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
def jac(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(1, len(A | B))
treino = [norm(json.loads(l)["question"]) for l in open(D / "questoes.jsonl")]
bench = [json.loads(l) for l in open(D / "benchmark_v2_raw220.jsonl")]
EN = {"what", "how", "which", "who", "when", "where", "why", "the", "in", "a"}
mantidos, removidos, rem_en = [], 0, 0
for b in bench:
    nb = norm(b["question"])
    # sementes do docentesDC incluem material em inglês → o professor pode gerar pergunta EN;
    # o benchmark é PT (validação pré-submissão: 14/800 EN no pool de treino)
    if nb.split()[0] in EN:
        rem_en += 1
        continue
    if nb in treino or max((jac(nb, t) for t in treino), default=0) > 0.7:
        removidos += 1
        continue
    mantidos.append(b)
print(f"   pós-filtro: {removidos} quase-duplicatas + {rem_en} em inglês removidas, {len(mantidos)} restantes")
for src, tag in [("DOM-PI", "dompi"), ("docentesDC", "docentes")]:
    itens = [b for b in mantidos if b["source"] == src][:100]
    with open(D / f"benchmark_{tag}_n100.jsonl", "w") as f:
        for i, b in enumerate(itens):
            b["id"] = f"bx{tag[:3]}{i:03d}"
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"   bench {tag}: {len(itens)} perguntas (alvo 100)")
PYEOF

# 3) Reavalia os 6 modelos existentes, cada especialista no benchmark do seu tópico
echo ">> [3/3] Avaliando bases + especialistas nos benchmarks n100..."
for tag in dompi docentes; do
  $PY $SCR/avaliar_destilacao.py \
      --benchmark "$DADOS/benchmark_${tag}_n100.jsonl" \
      --out "$RES/avaliacao_${tag}_n100.json" \
      --model "base_05=Qwen/Qwen2.5-0.5B" --model "base_15=Qwen/Qwen2.5-1.5B" \
      --model "esp_${tag}_0.5b=$MODELOS/esp_${tag}_0.5b" \
      --model "esp_${tag}_1.5b=$MODELOS/esp_${tag}_1.5b"
done
echo "=== q4-bench200 concluído → $RES/avaliacao_{dompi,docentes}_n100.json ==="
