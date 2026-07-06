#!/bin/bash
# q4_orchestrator.sh — mantém as GPUs SEMPRE utilizadas: vigia o SLURM e, quando nenhum job nosso
# está na fila (GPUs livres), submete o próximo passo PENDENTE do pipeline. Persistente (nohup),
# sobrevive à desconexão. Idempotente: pula passos cujo "marker" (saída) já existe; não re-submete
# passos que falharam (registra em .orch_tried/). Re-lê o pipeline a cada ciclo → aceita novos passos.
#
# Uso (no cluster):  cd ~/building-a-LLM && nohup bash trabalho-final/Q4-destilacao/scripts/q4_orchestrator.sh >/dev/null 2>&1 &
# Pipeline: trabalho-final/Q4-destilacao/scripts/q4_pipeline.txt  (linha: <sbatch> | <glob-marker>)

cd "$HOME/building-a-LLM"
PIPE=trabalho-final/Q4-destilacao/scripts/q4_pipeline.txt
LOG=logs/q4_orchestrator.log
mkdir -p logs .orch_tried
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "=== orquestrador iniciado (PID $$) ==="

while true; do
  # GPUs ocupadas por job nosso (R ou PD)? espera.
  if squeue -u aluno_matheus -h -o "%j %t" 2>/dev/null | grep -qiE "q4"; then
    sleep 30; continue
  fi

  # acha o próximo passo pendente (marker ausente e não-falho)
  ns=""; nm=""; nh=""
  while IFS='|' read -r sb marker; do
    sb="$(echo "$sb" | xargs)"; marker="$(echo "$marker" | xargs)"
    [ -z "$sb" ] && continue
    case "$sb" in \#*) continue ;; esac
    [ -n "$marker" ] && ls $marker >/dev/null 2>&1 && continue          # já concluído
    h="$(echo "$sb" | md5sum | cut -c1-12)"
    [ -f ".orch_tried/$h" ] && continue                                  # já falhou antes
    ns="$sb"; nm="$marker"; nh="$h"; break
  done < "$PIPE"

  if [ -z "$ns" ]; then sleep 60; continue; fi                          # nada pendente; segue vigiando

  log "GPUs livres → submetendo: $ns"
  sinfo -h -o '  nodo=%n estado=%t gres=%G' >> "$LOG" 2>/dev/null
  jid="$(sbatch "$ns" 2>&1 | grep -oE '[0-9]+' | head -1)"
  if [ -z "$jid" ]; then log "  ERRO ao submeter $ns — pulando"; touch ".orch_tried/$nh"; continue; fi
  log "  job $jid submetido; aguardando conclusão"
  while squeue -j "$jid" -h -o %t 2>/dev/null | grep -q .; do sleep 30; done
  if [ -n "$nm" ] && ls $nm >/dev/null 2>&1; then
    log "  CONCLUÍDO $ns (marker: $nm)"
  else
    touch ".orch_tried/$nh"; log "  FALHOU $ns (sem marker $nm) — não re-submete"
  fi
done
