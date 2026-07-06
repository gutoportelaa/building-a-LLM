#!/bin/bash
#SBATCH --job-name=dompi-relatorio-q2
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=treino/log_relatorio_q2_%j.txt
#SBATCH --error=treino/log_relatorio_q2_%j.txt

# Sem GPU — só lê JSONs e gera HTML
cd "$SLURM_SUBMIT_DIR"

echo "=== Gerando relatorio_questao2.html ==="
echo "Job ID: $SLURM_JOB_ID | Data: $(date)"

.venv/bin/python3 avaliacao/gerar_relatorio_q2.py

echo "=== Concluído: $(date) ==="
echo "Arquivos de resultado disponíveis:"
ls -lh avaliacao/resultados_q2_*.json 2>/dev/null || echo "  (nenhum encontrado)"
ls -lh relatorio_questao2.html 2>/dev/null || echo "  (HTML não gerado)"
