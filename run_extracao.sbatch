#!/bin/bash
# ============================================================================
# run_extracao.sbatch — extração de um território no cluster SLURM (lab UFPI)
# ----------------------------------------------------------------------------
# Submissão (no head/slurm-master, dentro de ~/building-a-LLM):
#     sbatch run_extracao.sbatch tabuleiros_alto_parnaiba
#     sbatch --time=06:00:00 run_extracao.sbatch vale_do_sambito
#
# Fonte dos PDFs = VARREDURA da pasta territorios/<slug>/pdfs (cobre TODOS os
# PDFs presentes; o download_manifest.json do projeto está incompleto). O
# município/entidade é inferido da estrutura .../pdfs/<municipio>/<entidade>/.
#
# Idempotente/resumível: o orquestrador faz checkpoint a cada 25 docs e ao
# receber SIGTERM. Se o job bater o time-limit (SLURM manda SIGTERM 120s antes
# do SIGKILL via --signal abaixo), o progresso é salvo.
#
# AUTO-ENCADEAMENTO: ao final, se a extração foi INTERROMPIDA por walltime (e não
# concluída), o próprio job RE-SUBMETE a si mesmo (resume). Assim a extração anda
# sozinha entre cortes de walltime até 100%, re-adquirindo GPU a cada rodada.
# Para uma corrida só (sem encadear), submeta com  AUTO_RESUBMIT=0:
#     sbatch --export=ALL,AUTO_RESUBMIT=0 run_extracao.sbatch <slug>
#
# 1 GPU (gpu:l4:1) + walltime de 8h são DE PROPÓSITO: pedir 2 GPUs/2-dias trava na
# fila atrás de jobs grandes; com 1 GPU/8h o SLURM faz BACKFILL na 1ª L4 ociosa e
# inicia já. Ganho de 2 GPUs no loop sequencial é marginal — não compensa esperar.
# As interrupções/perdas ficam em extraidos/<slug>/eventos_extracao.ndjson.
# ============================================================================
#SBATCH --job-name=dompi-extrai
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:1           # 1 GPU = backfilla na L4 ociosa; 2 trava na fila
#SBATCH --mem=28G                 # ARMADILHA: sem isto o SLURM reserva a RAM do nó inteiro.
                                  # 28G (não 32) p/ caberem 2 jobs no nó (RealMemory 62,9G):
                                  # 2×32=64>62,9 não coexistem; 2×28=56 sim → paralelismo real.
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00           # janela curta habilita backfill; checkpoint cobre cortes
#SBATCH --signal=B:TERM@120       # SIGTERM 120s antes do SIGKILL → janela p/ flush
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail
TERRITORIO="${1:?uso: sbatch run_extracao.sbatch <slug_territorio>}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "extraidos/$TERRITORIO"

echo "== Job $SLURM_JOB_ID em $(hostname) | território=$TERRITORIO =="
echo "PDFs a varrer: $(find "territorios/$TERRITORIO/pdfs" -name '*.pdf' 2>/dev/null | wc -l)"

# Atribuição de GPU adaptativa ao que o SLURM alocou: com 2 GPUs, paddle=0 e
# docling=1 (sem contenção); com 1 GPU, ambos no índice 0 (compartilham a L4).
# NÃO usar nvidia-smi: o NVML está QUEBRADO no gpunode01 (compute CUDA ok) e
# nvidia-smi retorna != 0, o que mataria o script sob `set -e`/`pipefail`.
# Contamos pelas variáveis do SLURM (índices relativos a CUDA_VISIBLE_DEVICES).
NGPU="${SLURM_GPUS_ON_NODE:-}"
if [ -z "$NGPU" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NGPU=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    else
        NGPU=1
    fi
fi
DOCLING_GPU=0
[ "${NGPU:-1}" -ge 2 ] && DOCLING_GPU=1
echo "GPUs alocadas: $NGPU (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}) → paddle=GPU0 docling=GPU$DOCLING_GPU"

# Roda o Python em BACKGROUND e encaminha o SIGTERM do SLURM a ele (não dá p/ usar
# 'exec' aqui porque precisamos da lógica de auto-resubmit DEPOIS do Python sair).
# IMPORTANTE: ./.venv/bin/python direto (NUNCA `uv run`, que reinstala paddle no
# .venv e quebra o torch — ver docs/BENCHMARK_OCR.md).
EVENTOS="extraidos/$TERRITORIO/eventos_extracao.ndjson"

./.venv/bin/python -m dompi_scraper.orquestrador_extracao \
    --pdfs-dir      "territorios/$TERRITORIO/pdfs" \
    --output-dir    "extraidos/$TERRITORIO/datalake" \
    --registry-dir  "extraidos/$TERRITORIO" \
    --corpus-output "extraidos/$TERRITORIO/corpus_$TERRITORIO.jsonl" \
    --territorio    "$TERRITORIO" \
    --gpu-paddle 0 \
    --gpu-docling "$DOCLING_GPU" &
PYPID=$!

# Ao receber SIGTERM (corte de walltime), repassa ao Python p/ ele fazer o flush.
trap 'echo "[sbatch] SIGTERM → repassando ao python (pid $PYPID)"; kill -TERM "$PYPID" 2>/dev/null' TERM

# Espera o Python terminar DE FATO. Se um sinal interromper o `wait`, o trap roda
# e o laço re-entra no wait até o processo realmente sair (após o flush).
RC=0
while kill -0 "$PYPID" 2>/dev/null; do
    wait "$PYPID" && RC=$? || RC=$?
done
trap - TERM
echo "[sbatch] python encerrou com RC=$RC"

# --- Auto-resubmit encadeado -------------------------------------------------
# Re-submete SÓ se ESTE job foi interrompido por walltime (último evento =
# 'interrupcao' com o nosso job_id). Se o orquestrador varreu a lista inteira,
# não há 'interrupcao' nova → extração concluída → para o encadeamento.
if [ "${AUTO_RESUBMIT:-1}" = "1" ]; then
    ULT=$(tail -1 "$EVENTOS" 2>/dev/null || true)
    if printf '%s' "$ULT" | grep -q '"evento": "interrupcao"' \
       && printf '%s' "$ULT" | grep -q "\"job_id\": \"$SLURM_JOB_ID\""; then
        echo "[sbatch] extração INCOMPLETA (cortada por walltime) → re-submetendo resume"
        sbatch "$SLURM_SUBMIT_DIR/run_extracao.sbatch" "$TERRITORIO"
    else
        echo "[sbatch] extração CONCLUÍDA (sem interrupção pendente) → fim do encadeamento"
    fi
fi
exit "$RC"
