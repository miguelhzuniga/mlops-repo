#!/bin/bash

# Script para desmontar los servicios de FastAPI y Gradio en Kubernetes con microk8s
# Fecha: Mayo 2025

echo "====================================================================="
echo "Iniciando desmontaje de servicios del Predictor de Diabetes..."
echo "====================================================================="

# Definir variables
NAMESPACE="mlops-project"
HOST_IP="10.43.101.206"

echo "Namespace Kubernetes: $NAMESPACE"
echo

# Verificar que microk8s está disponible
if ! command -v microk8s &> /dev/null; then
    echo "microk8s no está instalado. Por favor, instálalo e inténtalo de nuevo."
    exit 1
fi

# Verificar que el namespace existe
if ! microk8s kubectl get namespace | grep -q "$NAMESPACE"; then
    echo "El namespace $NAMESPACE no existe. No hay nada que desmontar."
    exit 1
fi

# 1. Listar los recursos antes de eliminarlos
echo "Recursos actuales en el namespace $NAMESPACE:"
echo "-----------------------------------"
echo "Pods:"
microk8s kubectl get pods -n $NAMESPACE | grep -E 'fastapi-diabetes|gradio-diabetes'
echo "Servicios:"
microk8s kubectl get services -n $NAMESPACE | grep -E 'fastapi-diabetes|gradio-diabetes'
echo "Deployments:"
microk8s kubectl get deployments -n $NAMESPACE | grep -E 'fastapi-diabetes|gradio-diabetes'

# 2. Eliminar los servicios de Gradio
echo
echo "Eliminando servicio Gradio..."
echo "---------------------------------"
# Intentar eliminar el deployment de Gradio
if microk8s kubectl get deployment gradio-diabetes -n $NAMESPACE &>/dev/null; then
    echo "Eliminando deployment de Gradio..."
    microk8s kubectl delete deployment gradio-diabetes -n $NAMESPACE
    
    if [ $? -eq 0 ]; then
        echo "Deployment de Gradio eliminado correctamente."
    else
        echo "Error al eliminar el deployment de Gradio."
    fi
else
    echo "No se encontró el deployment de Gradio."
fi

# Intentar eliminar el servicio de Gradio
if microk8s kubectl get service gradio-diabetes-service -n $NAMESPACE &>/dev/null; then
    echo "Eliminando servicio de Gradio..."
    microk8s kubectl delete service gradio-diabetes-service -n $NAMESPACE
    
    if [ $? -eq 0 ]; then
        echo "Servicio de Gradio eliminado correctamente."
    else
        echo "Error al eliminar el servicio de Gradio."
    fi
else
    echo "No se encontró el servicio de Gradio."
fi

# 3. Eliminar los servicios de FastAPI
echo
echo "Eliminando servicio FastAPI..."
echo "---------------------------------"
# Intentar eliminar el deployment de FastAPI
if microk8s kubectl get deployment fastapi-diabetes -n $NAMESPACE &>/dev/null; then
    echo "Eliminando deployment de FastAPI..."
    microk8s kubectl delete deployment fastapi-diabetes -n $NAMESPACE
    
    if [ $? -eq 0 ]; then
        echo "Deployment de FastAPI eliminado correctamente."
    else
        echo "Error al eliminar el deployment de FastAPI."
    fi
else
    echo "No se encontró el deployment de FastAPI."
fi

# Intentar eliminar el servicio de FastAPI
if microk8s kubectl get service fastapi-diabetes-service -n $NAMESPACE &>/dev/null; then
    echo "Eliminando servicio de FastAPI..."
    microk8s kubectl delete service fastapi-diabetes-service -n $NAMESPACE
    
    if [ $? -eq 0 ]; then
        echo "Servicio de FastAPI eliminado correctamente."
    else
        echo "Error al eliminar el servicio de FastAPI."
    fi
else
    echo "No se encontró el servicio de FastAPI."
fi

# 4. Verificar que los servicios han sido desmontados
echo
echo "Verificando que los servicios han sido desmontados..."
echo "---------------------------------------"

echo "Esperando 10 segundos para que se completen las operaciones..."
#sleep 10

echo "Pods restantes relacionados con los servicios (no debería haber ninguno):"
microk8s kubectl get pods -n $NAMESPACE | grep -E 'fastapi-diabetes|gradio-diabetes' || echo "No se encontraron pods."

echo "Servicios restantes relacionados (no debería haber ninguno):"
microk8s kubectl get services -n $NAMESPACE | grep -E 'fastapi-diabetes|gradio-diabetes' || echo "No se encontraron servicios."

echo "Deployments restantes relacionados (no debería haber ninguno):"
microk8s kubectl get deployments -n $NAMESPACE | grep -E 'fastapi-diabetes|gradio-diabetes' || echo "No se encontraron deployments."

# 5. Opcionalmente, eliminar las imágenes Docker si ya no son necesarias
for image in "${IMAGES_TO_DELETE[@]}"; do
  if sudo docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${image}$"; then
    echo -e "${YELLOW} - Eliminando imagen ${image}${NC}"
    sudo docker rmi -f "${image}" >/dev/null 2>&1 && \
      echo -e "${GREEN}   ✔ Imagen eliminada: ${image}${NC}" || \
      echo -e "${RED}   ✖ No se pudo eliminar: ${image}${NC}"
  else
    echo -e "${YELLOW}   ⚠ Imagen no encontrada localmente: ${image}${NC}"
  fi
done

echo
echo "====================================================================="
echo "Desmontaje completado."
echo "Los servicios en http://$HOST_IP:30602 y http://$HOST_IP:30601 ya no deberían estar disponibles."
echo "====================================================================="