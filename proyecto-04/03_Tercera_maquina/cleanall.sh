#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}===== Limpiando Prometheus y Grafana de MicroK8s =====${NC}"

if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Por favor, ejecute como root o con sudo${NC}"
  exit 1
fi

if ! command -v microk8s >/dev/null 2>&1; then
  echo -e "${RED}MicroK8s no encontrado.${NC}"
  exit 1
fi

echo -e "${YELLOW}Verificando estado de MicroK8s...${NC}"
microk8s status --wait-ready

if ! command -v kubectl >/dev/null 2>&1; then
  echo -e "${YELLOW}Creando alias kubectl...${NC}"
  snap alias microk8s.kubectl kubectl
  echo -e "${GREEN}Alias kubectl creado${NC}"
fi

echo -e "${YELLOW}Eliminando recursos de Prometheus...${NC}"
microk8s kubectl delete -f ./monitoring/prometheus.yaml --ignore-not-found=true

echo -e "${YELLOW}Eliminando recursos de Grafana...${NC}"
microk8s kubectl delete -f ./monitoring/grafana.yaml --ignore-not-found=true

echo -e "${GREEN}===== Limpieza Completada =====${NC}"