#!/bin/bash
# Script para desmontar completamente MicroK8s y reiniciar desde cero
# Fecha: 1 de Mayo de 2025

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${RED}=== ATENCIÓN ===${NC}"
echo -e "Este script eliminará TODOS los recursos de Kubernetes y datos persistentes."
echo -e "Se perderán todos los datos almacenados en las bases de datos PostgreSQL, MinIO, etc."
read -p "¿Estás seguro que deseas continuar? (s/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    echo "Operación cancelada."
    exit 0
fi

echo -e "${YELLOW}=== Forzando eliminación del namespace mlops ===${NC}"
# Intentar eliminar el namespace normalmente
microk8s kubectl delete namespace mlops --timeout=30s

# Esperar un momento
sleep 3

# Verificar si sigue existiendo
if microk8s kubectl get namespace | grep -q mlops; then
    echo -e "${YELLOW}El namespace mlops está en estado de terminación. Eliminando recursos finalizers...${NC}"
    
    # Verificar si jq está instalado
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}jq no está instalado. Instalando jq...${NC}"
        # Intentar instalar jq de forma segura, ignorando errores de repositorios
        sudo apt-get --allow-unauthenticated -y install jq || sudo apt install -y --no-install-recommends jq
        
        # Verificar si la instalación fue exitosa
        if ! command -v jq &> /dev/null; then
            echo -e "${RED}Error: No se pudo instalar jq automáticamente.${NC}"
            echo -e "${YELLOW}Ejecutando instalación directa de jq...${NC}"
            # Último intento: descarga directa del paquete .deb para Ubuntu
            wget -O /tmp/jq.deb http://archive.ubuntu.com/ubuntu/pool/universe/j/jq/jq_1.6-2.1ubuntu3_amd64.deb && sudo dpkg -i /tmp/jq.deb
            
            if ! command -v jq &> /dev/null; then
                echo -e "${RED}Error: Todos los intentos fallaron. Instala jq manualmente:${NC}"
                echo "sudo apt install jq -y"
                exit 1
            else
                echo -e "${GREEN}jq instalado correctamente mediante descarga directa.${NC}"
            fi
        else
            echo -e "${GREEN}jq instalado correctamente.${NC}"
        fi
    fi
    
    # Extraer namespace y quitar finalizers
    microk8s kubectl get namespace mlops -o json > ns.json
    cat ns.json | jq 'del(.spec.finalizers)' > ns-modified.json
    
    # Iniciar proxy para usar API sin autenticación directa
    microk8s proxy &  # Proxy en segundo plano
    proxy_pid=$!
    sleep 3
    
    # Enviar solicitud al API para finalizar el namespace
    curl -s -H "Content-Type: application/json" -X PUT --data-binary @ns-modified.json \
        http://127.0.0.1:8001/api/v1/namespaces/mlops/finalize
    
    # Detener proxy y limpiar archivos temporales
    kill $proxy_pid
    rm ns.json ns-modified.json
    echo -e "${GREEN}Namespace mlops eliminado forzosamente.${NC}"
fi

echo -e "${YELLOW}=== Eliminando volúmenes persistentes ===${NC}"
microk8s kubectl delete pv --all

echo -e "${YELLOW}=== Limpiando directorios de datos ===${NC}"
sudo rm -rf /mnt/data/postgres-airflow
sudo rm -rf /mnt/data/airflow-dags
sudo rm -rf /mnt/data/postgres-mlflow
sudo rm -rf /mnt/data/minio

# Recrear
sudo mkdir -p /mnt/data/{postgres-mlflow,postgres-airflow,minio,airflow-dags}
sudo chmod -R 777 /mnt/data

echo -e "${YELLOW}=== Reiniciando MicroK8s ===${NC}"
microk8s stop
sleep 5
microk8s start
echo "Esperando a que MicroK8s se inicie completamente..."
sleep 10

echo -e "${YELLOW}=== Habilitando addons necesarios ===${NC}"
microk8s enable dns storage ingress

echo -e "${YELLOW}=== Verificando estado de MicroK8s ===${NC}"
microk8s status

echo -e "${YELLOW}=== Verificando que no hay recursos previos ===${NC}"
microk8s kubectl get all --all-namespaces | grep -vE "kube-system|ingress|default|container-registry"

echo -e "${GREEN}=== MicroK8s ha sido completamente reiniciado ===${NC}"
echo -e "Ahora puedes volver a desplegar tus recursos con:"
echo -e "bash deploy.sh"