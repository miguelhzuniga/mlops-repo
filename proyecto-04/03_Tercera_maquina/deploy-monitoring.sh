#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}===== Despliegue de Prometheus y Grafana en MicroK8s =====${NC}"


if ! command -v microk8s >/dev/null 2>&1; then
  echo -e "${YELLOW}MicroK8s no encontrado. Instalando...${NC}"
  snap install microk8s --classic
  usermod -a -G microk8s $SUDO_USER
  chown -f -R $SUDO_USER ~/.kube
  echo -e "${GREEN}MicroK8s instalado correctamente${NC}"
else
  echo -e "${GREEN}MicroK8s ya está instalado${NC}"
fi

echo -e "${YELLOW}Verificando estado de MicroK8s...${NC}"
microk8s status --wait-ready

if ! command -v kubectl >/dev/null 2>&1; then
  echo -e "${YELLOW}Creando alias kubectl...${NC}"
  snap alias microk8s.kubectl kubectl
  echo -e "${GREEN}Alias kubectl creado${NC}"
fi

NODE_NAME=$(hostname | tr '[:upper:]' '[:lower:]')
echo -e "${GREEN}Nombre del nodo: ${NODE_NAME}${NC}"

echo -e "${YELLOW}Etiquetando nodo...${NC}"
microk8s kubectl label node $NODE_NAME node-type=worker node-id=$NODE_NAME --overwrite

echo -e "${YELLOW}Creando namespace 'mlops-project' si no existe...${NC}"
microk8s kubectl create namespace mlops-project --dry-run=client -o yaml | microk8s kubectl apply -f -

if [ ! -f "./monitoring/prometheus.yaml" ] || [ ! -f "./monitoring/grafana.yaml" ]; then
  echo -e "${RED}Manifiestos prometheus.yaml o grafana.yaml no encontrados en ./monitoring${NC}"
  exit 1
fi

echo -e "${YELLOW}Desplegando Prometheus...${NC}"
microk8s kubectl apply -f ./monitoring/prometheus.yaml

echo -e "${YELLOW}Desplegando Grafana...${NC}"
microk8s kubectl apply -f ./monitoring/grafana.yaml

echo -e "${YELLOW}Esperando a que los pods de Prometheus y Grafana estén listos...${NC}"
microk8s kubectl rollout status deployment/prometheus -n mlops-project --timeout=90s
microk8s kubectl rollout status deployment/grafana -n mlops-project --timeout=90s

NODE_IP=$(hostname -I | awk '{print $1}')

echo -e "${GREEN}===== Despliegue Completo =====${NC}"
echo -e "${YELLOW}Accede a Prometheus en: http://${NODE_IP}:31090${NC}"
echo -e "${YELLOW}Accede a Grafana en: http://${NODE_IP}:31300${NC}"
echo -e "${GREEN}Credenciales de Grafana: admin / admin${NC}"
echo -e "${GREEN}=====================================${NC}"