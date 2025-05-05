#!/bin/bash
# deploy-services.sh - Script para unir un nodo al clúster MicroK8s y desplegar servicios ML

set -e  # Salir ante cualquier error

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 

MLFLOW_NODE_IP="" 
MINIO_NODE_IP=""  

echo -e "${YELLOW}===== Configuración de Nodo MicroK8s y Despliegue de Servicios =====${NC}"

# Verificar si se está ejecutando como root o con sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Por favor, ejecute como root o con sudo${NC}"
  exit 1
fi

# Verificar si MicroK8s está instalado
if ! command -v microk8s >/dev/null 2>&1; then
  echo -e "${YELLOW}MicroK8s no encontrado. Instalando...${NC}"
  snap install microk8s --classic
  usermod -a -G microk8s $SUDO_USER
  chown -f -R $SUDO_USER ~/.kube
  echo -e "${GREEN}MicroK8s instalado correctamente${NC}"
else
  echo -e "${GREEN}MicroK8s ya está instalado${NC}"
fi

# Asegurar que MicroK8s está funcionando
echo -e "${YELLOW}Verificando estado de MicroK8s...${NC}"
microk8s status --wait-ready

# Crear alias kubectl si no existe
if ! command -v kubectl >/dev/null 2>&1; then
  echo -e "${YELLOW}Creando alias kubectl...${NC}"
  snap alias microk8s.kubectl kubectl
  echo -e "${GREEN}Alias kubectl creado${NC}"
fi

# Obtener comando de unión del usuario
echo -e "${YELLOW}Ingrese el comando de unión del nodo principal (salida de microk8s add-node):${NC}"
read -p "> " JOIN_COMMAND

# Ejecutar el comando de unión
echo -e "${YELLOW}Uniendo al clúster...${NC}"
eval $JOIN_COMMAND


# Obtener el nombre del nodo
NODE_NAME=$(hostname)
echo -e "${GREEN}Nombre del nodo: ${NODE_NAME}${NC}"

# Etiquetar el nodo
echo -e "${YELLOW}Etiquetando el nodo...${NC}"
microk8s kubectl label node $NODE_NAME node-type=worker node-id=$NODE_NAME
echo -e "${GREEN}Nodo etiquetado correctamente${NC}"

# Crear namespace si no existe
echo -e "${YELLOW}Creando namespace 'mlops-project'...${NC}"
microk8s kubectl create namespace mlops-project --dry-run=client -o yaml | microk8s kubectl apply -f -

# Verificar si tenemos los archivos de despliegue
if [ ! -f "./api/fastapi-deployment.yaml" ] || [ ! -f "./streamlit/streamlit-deployment.yaml" ]; then
  echo -e "${RED}Archivos de despliegue no encontrados. Asegúrese de que existan en los directorios api y streamlit.${NC}"
  exit 1
fi

# Obtener la IP del nodo actual para FastAPI
FASTAPI_NODE_IP=$(hostname -I | awk '{print $1}')
echo -e "${GREEN}IP del nodo FastAPI: ${FASTAPI_NODE_IP}${NC}"
echo -e "${GREEN}IP del nodo MLflow: ${MLFLOW_NODE_IP}${NC}"
echo -e "${GREEN}IP del nodo MinIO: ${MINIO_NODE_IP}${NC}"

# Reemplazar las variables en los archivos YAML
echo -e "${YELLOW}Actualizando configuraciones con las IPs correctas...${NC}"
sed -i "s|\${MLFLOW_NODE_IP}|$MLFLOW_NODE_IP|g" ./api/fastapi-deployment.yaml
sed -i "s|\${MINIO_NODE_IP}|$MINIO_NODE_IP|g" ./api/fastapi-deployment.yaml
sed -i "s|\${FASTAPI_NODE_IP}|$FASTAPI_NODE_IP|g" ./streamlit/streamlit-deployment.yaml

# Construir e impulsar imágenes Docker
echo -e "${YELLOW}Construyendo e impulsando imágenes Docker...${NC}"

# Obtener URL del registro si no se proporciona a través de variable de entorno
if [ -z "$REGISTRY_URL" ]; then
  echo -e "${YELLOW}Ingrese la URL de su registro de contenedores (predeterminado: localhost:32000):${NC}"
  read -p "> " REGISTRY_URL
  REGISTRY_URL=${REGISTRY_URL:-localhost:32000}
  
  # Actualizar las URLs de imagen en los archivos de despliegue
  echo -e "${YELLOW}Actualizando URLs de imagen en archivos de despliegue...${NC}"
  sed -i "s|\${YOUR_REGISTRY}|$REGISTRY_URL|g" ./api/fastapi-deployment.yaml
  sed -i "s|\${YOUR_REGISTRY}|$REGISTRY_URL|g" ./streamlit/streamlit-deployment.yaml
fi

# Construir imagen FastAPI
echo -e "${YELLOW}Construyendo imagen FastAPI...${NC}"
docker build -t ${REGISTRY_URL}/fastapi-diabetes:latest ./api

# Construir imagen Streamlit
echo -e "${YELLOW}Construyendo imagen Streamlit...${NC}"
docker build -t ${REGISTRY_URL}/streamlit-diabetes:latest ./streamlit

# Impulsar imágenes al registro
echo -e "${YELLOW}Impulsando imágenes al registro...${NC}"
docker push ${REGISTRY_URL}/fastapi-diabetes:latest
docker push ${REGISTRY_URL}/streamlit-diabetes:latest

echo -e "${GREEN}Imágenes construidas e impulsadas correctamente${NC}"

# Desplegar servicios
echo -e "${YELLOW}Desplegando servicios en Kubernetes...${NC}"
microk8s kubectl apply -f ./api/fastapi-deployment.yaml
microk8s kubectl apply -f ./streamlit/streamlit-deployment.yaml

# Esperar a que los servicios estén listos
echo -e "${YELLOW}Esperando a que los servicios estén listos...${NC}"
microk8s kubectl rollout status deployment/fastapi-diabetes -n mlops-project
microk8s kubectl rollout status deployment/streamlit-diabetes -n mlops-project

# Obtener la IP del nodo
NODE_IP=$(hostname -I | awk '{print $1}')

echo -e "${GREEN}===== Despliegue Completo! =====${NC}"
echo -e "${GREEN}Servicios desplegados correctamente:${NC}"
echo -e "${YELLOW}Servicio FastAPI: http://${NODE_IP}:30501${NC}"
echo -e "${YELLOW}UI Streamlit: http://${NODE_IP}:30502${NC}"
echo -e "${YELLOW}Conexión a MLflow: http://${MLFLOW_NODE_IP}:30500${NC}"
echo -e "${YELLOW}Conexión a MinIO: http://${MINIO_NODE_IP}:9000${NC}"
echo /e "Estas son las maquinas conectadas al cluster"
microk8s kubectl get nodes
echo -e "${GREEN}=================================${NC}"