#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "====================================================================="
echo "Descargando y desplegando servicios desde Docker Hub..."
echo "====================================================================="

NAMESPACE="mlops-project"
FASTAPI_DIR="./fastapi"
GRADIO_DIR="./gradio"
HOST_IP=$(hostname -I | awk '{print $1}')
DOCKERHUB_USER="luisfrontuso10"

# Descargar imágenes más recientes
echo -e "${YELLOW}Descargando imágenes de Docker Hub...${NC}"
docker pull $DOCKERHUB_USER/fastapi-houses:latest
docker pull $DOCKERHUB_USER/gradio-houses:latest

# Limpiar despliegues anteriores
echo -e "${YELLOW}Limpiando despliegues anteriores...${NC}"
microk8s kubectl delete deployment fastapi-housing gradio-housing -n $NAMESPACE --ignore-not-found=true
microk8s kubectl delete service fastapi-housing-service gradio-housing-service -n $NAMESPACE --ignore-not-found=true
echo "Esperando 5 segundos para que los recursos se eliminen completamente..."
sleep 5

# Aplicar configuraciones
echo -e "${YELLOW}Aplicando configuraciones Kubernetes...${NC}"
microk8s kubectl apply -f $FASTAPI_DIR/fastapi-deployment.yaml -n $NAMESPACE
microk8s kubectl apply -f $FASTAPI_DIR/fastapi-service.yaml -n $NAMESPACE
microk8s kubectl apply -f $GRADIO_DIR/gradio-deployment.yaml -n $NAMESPACE
microk8s kubectl apply -f $GRADIO_DIR/gradio-service.yaml -n $NAMESPACE

echo "Esperando 10 segundos para que los servicios se inicien..."
sleep 10

echo -e "${YELLOW}Estado de los pods:${NC}"
microk8s kubectl get pods -n $NAMESPACE

# Obtener y mostrar los NodePorts de los servicios
echo
echo -e "${GREEN}NodePorts de los servicios:${NC}"
echo "-----------------------------------"
FASTAPI_NODEPORT=$(microk8s kubectl get service fastapi-housing-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
GRADIO_NODEPORT=$(microk8s kubectl get service gradio-housing-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
echo "FastAPI: http://$HOST_IP:$FASTAPI_NODEPORT"
echo "Gradio: http://$HOST_IP:$GRADIO_NODEPORT"
echo "====================================================================="
echo -e "${GREEN}Despliegue completado.${NC}"
echo "Acceda a los servicios en:"
echo "FastAPI: http://$HOST_IP:$FASTAPI_NODEPORT"
echo "Gradio: http://$HOST_IP:$GRADIO_NODEPORT"
echo "====================================================================="