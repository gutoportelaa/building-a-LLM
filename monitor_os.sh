#!/bin/bash
# Script de monitoramento do sistema operacional (WSL)
# Registra consumo de RAM, CPU, processos mais pesados e logs do kernel (dmesg)

LOG_FILE="os_monitor_crash.log"

echo "==========================================================" > "$LOG_FILE"
echo "Iniciando monitoramento de SO. Data: $(date)" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"
echo "Versão Kernel: $(uname -a)" >> "$LOG_FILE"
echo "Memória Total: $(free -h)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Apenas prossegue

while true; do
    echo "--- [ $(date '+%Y-%m-%d %H:%M:%S') ] ---" >> "$LOG_FILE"
    
    # Memória
    free -m | grep Mem >> "$LOG_FILE"
    
    # Top 5 processos usando memória
    echo "[Top 5 processos de RAM]" >> "$LOG_FILE"
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 6 >> "$LOG_FILE"
    
    # Top 5 processos usando CPU
    # echo "[Top 5 processos de CPU]" >> "$LOG_FILE"
    # ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 6 >> "$LOG_FILE"
    
    # Captura novos eventos de Kernel (especialmente OOM Killer)
    KERNEL_LOGS=$(dmesg | tail -n 15)
    if [[ -n "$KERNEL_LOGS" ]]; then
        echo "[Eventos do Kernel (dmesg)]" >> "$LOG_FILE"
        echo "$KERNEL_LOGS" >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
    sleep 1
done
