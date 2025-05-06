#!/bin/bash

# Colores para mejor visualización
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "====================================================================="
echo "Construyendo, etiquetando y desplegando servicios..."
echo "====================================================================="

# Definir variables
NAMESPACE="mlops-project"
FASTAPI_DIR="./fastapi"
GRADIO_DIR="./gradio"
HOST_IP=$(hostname -I | awk '{print $1}')
DOCKERHUB_USER="camilosvel"

# Construir y subir la imagen FastAPI
echo -e "${YELLOW}Construyendo imagen FastAPI...${NC}"
sudo docker build -t api $FASTAPI_DIR
echo -e "${YELLOW}Etiquetando imagen FastAPI...${NC}"
sudo docker tag api $DOCKERHUB_USER/fastapi-diabetes:latest
echo -e "${YELLOW}Subiendo imagen FastAPI a Docker Hub...${NC}"
sudo docker push $DOCKERHUB_USER/fastapi-diabetes:latest

# Construir y subir la imagen Gradio
echo -e "${YELLOW}Construyendo imagen Gradio...${NC}"
sudo docker build -t gradio $GRADIO_DIR
echo -e "${YELLOW}Etiquetando imagen Gradio...${NC}"
sudo docker tag gradio $DOCKERHUB_USER/gradio-diabetes:latest
echo -e "${YELLOW}Subiendo imagen Gradio a Docker Hub...${NC}"
sudo docker push $DOCKERHUB_USER/gradio-diabetes:latest

# Limpiar despliegues anteriores
echo -e "${YELLOW}Limpiando despliegues anteriores...${NC}"
microk8s kubectl delete deployment fastapi-diabetes gradio-diabetes -n $NAMESPACE --ignore-not-found=true
microk8s kubectl delete service fastapi-diabetes-service gradio-diabetes-service -n $NAMESPACE --ignore-not-found=true
echo "Esperando 5 segundos para que los recursos se eliminen completamente..."
sleep 5

# Habilitar el registro de MicroK8s si no está habilitado
echo "Asegurando que el registro local está habilitado..."
microk8s enable registry

# Verificar imágenes
echo "Verificando imágenes en Docker..."
docker images | grep -E 'fastapi|gradio'

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
FASTAPI_NODEPORT=$(microk8s kubectl get service fastapi-diabetes-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
GRADIO_NODEPORT=$(microk8s kubectl get service gradio-diabetes-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
echo "FastAPI: http://$HOST_IP:$FASTAPI_NODEPORT"
echo "Gradio: http://$HOST_IP:$GRADIO_NODEPORT"
echo "====================================================================="
echo -e "${GREEN}Despliegue completado.${NC}"
echo "Acceda a los servicios en:"
echo "FastAPI: http://$HOST_IP:$FASTAPI_NODEPORT"
echo "Gradio: http://$HOST_IP:$GRADIO_NODEPORT"
echo "====================================================================="