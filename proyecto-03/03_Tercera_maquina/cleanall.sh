#!/bin/bash
# cleanall.sh - Elimina Locust, Prometheus y Grafana del clúster de MicroK8s

set -e  # Salir ante cualquier error

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}===== Limpiando Locust, Prometheus y Grafana de MicroK8s =====${NC}"

# Verificar si se está ejecutando como root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Por favor, ejecute como root o con sudo${NC}"
  exit 1
fi

# Verificar MicroK8s
if ! command -v microk8s >/dev/null 2>&1; then
  echo -e "${RED}MicroK8s no encontrado. Asegúrate de tener MicroK8s instalado y funcionando.${NC}"
  exit 1
fi

# Verificar estado de MicroK8s
echo -e "${YELLOW}Verificando estado de MicroK8s...${NC}"
microk8s status --wait-ready

# Crear alias para kubectl si no existe
if ! command -v kubectl >/dev/null 2>&1; then
  echo -e "${YELLOW}Creando alias kubectl...${NC}"
  snap alias microk8s.kubectl kubectl
  echo -e "${GREEN}Alias kubectl creado${NC}"
fi

# Eliminar cualquier recurso de Locust, Prometheus y Grafana
echo -e "${YELLOW}Eliminando recursos de Locust...${NC}"
microk8s kubectl delete -f ./locust/locust.yaml

echo -e "${YELLOW}Eliminando recursos de Prometheus...${NC}"
microk8s kubectl delete -f ./monitoring/prometheus.yaml

echo -e "${YELLOW}Eliminando recursos de Grafana...${NC}"
microk8s kubectl delete -f ./monitoring/grafana.yaml


echo -e "${GREEN}===== Limpieza Completada =====${NC}"
