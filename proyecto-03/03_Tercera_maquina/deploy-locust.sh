#!/bin/bash
# deploy-locust.sh - Despliega Locust en MicroK8s para pruebas de carga

set -e

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}===== Despliegue de Locust en MicroK8s =====${NC}"

# Verificar si se está ejecutando como root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Por favor, ejecute como root o con sudo${NC}"
  exit 1
fi

# Verificar microk8s
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

# Crear alias para kubectl si no existe
if ! command -v kubectl >/dev/null 2>&1; then
  echo -e "${YELLOW}Creando alias para kubectl...${NC}"
  snap alias microk8s.kubectl kubectl
  echo -e "${GREEN}Alias kubectl creado${NC}"
fi

# Comprobar si el nodo ya está unido
if microk8s kubectl get nodes &>/dev/null; then
    echo -e "${YELLOW}Este nodo ya está unido a un clúster MicroK8s.${NC}"
else
    echo -e "${YELLOW}Ingrese el comando de unión del nodo principal (salida de microk8s add-node):${NC}"
    read -p "> " JOIN_COMMAND
    eval "microk8s join ${JOIN_COMMAND}"
fi

# Obtener nombre del nodo
NODE_NAME=$(hostname | tr '[:upper:]' '[:lower:]')
echo -e "${GREEN}Nombre del nodo: ${NODE_NAME}${NC}"

# Etiquetar nodo si es necesario
echo -e "${YELLOW}Etiquetando nodo...${NC}"
microk8s kubectl label node $NODE_NAME node-type=worker node-id=$NODE_NAME --overwrite

# Crear namespace
echo -e "${YELLOW}Creando namespace 'mlops-project' si no existe...${NC}"
microk8s kubectl create namespace mlops-project --dry-run=client -o yaml | microk8s kubectl apply -f -

# Verificar que el manifiesto esté presente
if [ ! -f "./locust/locust.yaml" ]; then
  echo -e "${RED}Manifiesto locust.yaml no encontrado en ./locust${NC}"
  exit 1
fi

# Aplicar manifiesto
echo -e "${YELLOW}Desplegando Locust...${NC}"
microk8s kubectl apply -f ./locust/locust.yaml

# Esperar a que esté listo
echo -e "${YELLOW}Esperando a que Locust esté listo...${NC}"
microk8s kubectl rollout status deployment/locust -n mlops-project --timeout=90s

# Obtener IP y mostrar URL de acceso
NODE_IP=$(hostname -I | awk '{print $1}')
echo -e "${GREEN}===== Locust Desplegado =====${NC}"
echo -e "${YELLOW}Accede a Locust en: http://${NODE_IP}:31000${NC}"
echo -e "${GREEN}=====================================${NC}"
