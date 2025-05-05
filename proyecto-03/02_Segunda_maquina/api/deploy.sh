#!/bin/bash

echo "====================================================================="
echo "Desplegando servicios usando registro local de Docker..."
echo "====================================================================="

# Definir variables
NAMESPACE="mlops-project"
FASTAPI_DIR="./fastapi"
GRADIO_DIR="./gradio"
HOST_IP=$(hostname -I | awk '{print $1}')

# Limpiar despliegues anteriores
echo "Limpiando despliegues anteriores..."
microk8s kubectl delete deployment fastapi-diabetes gradio-diabetes -n $NAMESPACE --ignore-not-found=true
microk8s kubectl delete service fastapi-diabetes-service gradio-diabetes-service -n $NAMESPACE --ignore-not-found=true
echo "Esperando 5 segundos para que los recursos se eliminen completamente..."
sleep 5

# Habilitar el registro de MicroK8s si no está habilitado
echo "Asegurando que el registro local está habilitado..."
microk8s enable registry

DOCKERHUB_USER="camilosvel"

echo -e "${YELLOW}Obteniendo imágenes desde Docker Hub...${NC}"
docker pull ${DOCKERHUB_USER}/fastapi-diabetes:latest
docker pull ${DOCKERHUB_USER}/streamlit-diabetes:latest

echo -e "${GREEN}Imágenes traidas desde Dockerhub correctamente${NC}"

# Verificar imágenes
echo "Verificando imágenes en Docker..."
docker images | grep -E 'fastapi|gradio'

# Aplicar configuraciones
echo "Aplicando configuraciones Kubernetes..."

microk8s kubectl apply -f $FASTAPI_DIR/fastapi-deployment.yaml -n $NAMESPACE
microk8s kubectl apply -f $FASTAPI_DIR/fastapi-service.yaml -n $NAMESPACE

microk8s kubectl apply -f $GRADIO_DIR/gradio-deployment.yaml -n $NAMESPACE
microk8s kubectl apply -f $GRADIO_DIR/gradio-service.yaml -n $NAMESPACE

echo "Esperando 30 segundos para que los servicios se inicien..."
sleep 30

echo "Estado de los pods:"
microk8s kubectl get pods -n $NAMESPACE

# Obtener y mostrar los NodePorts de los servicios
echo
echo "NodePorts de los servicios:"
echo "-----------------------------------"
FASTAPI_NODEPORT=$(microk8s kubectl get service fastapi-diabetes-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
GRADIO_NODEPORT=$(microk8s kubectl get service gradio-diabetes-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')

echo "FastAPI: http://$HOST_IP:$FASTAPI_NODEPORT"
echo "Gradio: http://$HOST_IP:$GRADIO_NODEPORT"

echo "====================================================================="
echo "Despliegue completado."
echo "Acceda a los servicios en:"
echo "FastAPI: http://$HOST_IP:$FASTAPI_NODEPORT"
echo "Gradio: http://$HOST_IP:$GRADIO_NODEPORT"
echo "====================================================================="