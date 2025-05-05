#!/bin/bash
# deploy-monitoring.sh - Despliega Prometheus y Grafana en MicroK8s

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}===== Despliegue de Prometheus y Grafana en MicroK8s =====${NC}"

# Verificar si se ejecuta como root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Por favor, ejecute como root o con sudo${NC}"
  exit 1
fi

echo -e "${YELLOW}Verificando estado de MicroK8s...${NC}"
microk8s status --wait-ready

# Crear namespace si no existe
echo -e "${YELLOW}Creando namespace 'mlops-project' si no existe...${NC}"
microk8s kubectl create namespace mlops-project --dry-run=client -o yaml | microk8s kubectl apply -f -

# Verificar existencia de manifiestos
if [ ! -f "./monitoring/prometheus.yaml" ] || [ ! -f "./monitoring/grafana.yaml" ]; then
  echo -e "${RED}Manifiestos prometheus.yaml o grafana.yaml no encontrados en ./monitoring${NC}"
  exit 1
fi

echo -e "${YELLOW}Desplegando Prometheus...${NC}"
microk8s kubectl apply -f ./monitoring/prometheus.yaml

echo -e "${YELLOW}Desplegando Grafana...${NC}"
microk8s kubectl apply -f ./monitoring/grafana.yaml

echo -e "${YELLOW}Esperando a que los pods estén listos...${NC}"
microk8s kubectl rollout status deployment/prometheus -n mlops-project --timeout=90s
microk8s kubectl rollout status deployment/grafana -n mlops-project --timeout=90s

# Mostrar IPs
NODE_IP=$(hostname -I | awk '{print $1}')

echo -e "${GREEN}===== Monitoreo Desplegado =====${NC}"
echo -e "${YELLOW}Prometheus: http://${NODE_IP}:31090${NC}"
echo -e "${YELLOW}Grafana:    http://${NODE_IP}:31300${NC}"
echo -e "${GREEN}Credenciales por defecto de Grafana: admin / admin${NC}"
echo -e "${GREEN}=========================================${NC}"
