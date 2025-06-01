#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "====================================================================="
echo -e "${BLUE} Desplegando Argo CD en Kubernetes${NC}"
echo "====================================================================="

if ! command -v microk8s >/dev/null 2>&1; then
    echo " MicroK8s no está disponible"
    exit 1
fi

echo -e "${YELLOW} Instalando Argo CD...${NC}"

microk8s kubectl create namespace argocd --dry-run=client -o yaml | microk8s kubectl apply -f -
microk8s kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

echo -e "${YELLOW} Aplicando configuraciones personalizadas...${NC}"
microk8s kubectl apply -f install.yaml

echo -e "${YELLOW} Esperando a que Argo CD esté listo...${NC}"
microk8s kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=argocd-server -n argocd --timeout=300s

echo -e "${YELLOW} Instalando Argo CD Image Updater...${NC}"
microk8s kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj-labs/argocd-image-updater/stable/manifests/install.yaml
microk8s kubectl apply -f image-updater.yaml

echo -e "${YELLOW} Creando Applications...${NC}"
microk8s kubectl apply -f applications/

ARGOCD_PASSWORD=$(microk8s kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
NODE_IP=$(microk8s kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')

echo ""
echo "====================================================================="
echo -e "${GREEN} Argo CD desplegado exitosamente!${NC}"
echo "====================================================================="
echo -e " Argo CD UI: ${BLUE}https://$NODE_IP:30008${NC}"
echo -e " Usuario: ${YELLOW}admin${NC}"
echo -e " Password: ${YELLOW}$ARGOCD_PASSWORD${NC}"
echo ""
echo -e "${GREEN}Applications creadas:${NC}"
echo "• mlflow-app (MLflow + PostgreSQL + MinIO)"
echo "• api-services-app (FastAPI + Gradio)"  
echo "• monitoring-app (Prometheus + Grafana)"
echo ""
echo -e "${YELLOW} Image Updater detectará nuevas imágenes cada 2 minutos${NC}"
echo "====================================================================="