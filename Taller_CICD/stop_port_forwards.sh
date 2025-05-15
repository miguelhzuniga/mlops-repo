#!/bin/bash
if [ -d "port_forward_logs" ]; then
    for pid_file in port_forward_logs/*.pid; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            SERVICE=$(basename "$pid_file" .pid)
            if ps -p $PID > /dev/null; then
                echo "Deteniendo port-forward para $SERVICE (PID: $PID)..."
                kill $PID
            else
                echo "El proceso de port-forward para $SERVICE ya no está en ejecución"
            fi
        fi
    done
    echo "Todos los port-forwards detenidos"
else
    echo "No se encontró información de port-forwarding"
fi
