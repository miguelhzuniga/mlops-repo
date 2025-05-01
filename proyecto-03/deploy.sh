#!/bin/bash

# Script para desplegar MLflow y Airflow en una máquina con MicroK8s
# Autor: Luis Frontuso, Miguel Zuñiga, Camilo
# Fecha: 1 mayo 2025

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para instalar jq si no está disponible
install_jq() {
    echo -e "${YELLOW}jq no está instalado. Instalando jq...${NC}"
    # Intentar primero con apt-get sin actualizar repos
    sudo apt-get install -y --no-install-recommends jq || \
    # Si falla, probar con --allow-unauthenticated
    sudo apt-get install -y --allow-unauthenticated jq || \
    # Como último recurso, descargar directamente el paquete
    (wget -O /tmp/jq.deb http://archive.ubuntu.com/ubuntu/pool/universe/j/jq/jq_1.6-2.1ubuntu3_amd64.deb && \
     sudo dpkg -i /tmp/jq.deb && \
     rm -f /tmp/jq.deb)
    
    # Verificar la instalación
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: No se pudo instalar jq. Requiere instalación manual.${NC}"
        exit 1
    fi
    echo -e "${GREEN}jq instalado correctamente.${NC}"
}

# Detectar si estamos usando MicroK8s
USE_MICROK8S=false
if command -v microk8s &> /dev/null; then
    USE_MICROK8S=true
    # Alias para kubectl si usamos MicroK8s
    KUBECTL="microk8s kubectl"
else
    KUBECTL="kubectl"
fi

echo -e "${GREEN}=== MLOps Platform Deployment con MicroK8s ===${NC}"

# Verificar dependencias necesarias
if ! command -v jq &> /dev/null; then
    install_jq
fi

# Verificar que MicroK8s está instalado y en ejecución
if [ "$USE_MICROK8S" = true ]; then
    if ! microk8s status | grep "microk8s is running" &> /dev/null; then
        echo -e "${YELLOW}MicroK8s no está en ejecución. Intentando iniciar...${NC}"
        microk8s start
        sleep 10
    fi
    
    # Habilitar addons necesarios
    echo -e "${YELLOW}Habilitando addons necesarios de MicroK8s...${NC}"
    microk8s enable dns storage ingress
    
    # Verificar y habilitar Helm si no está habilitado
    if ! microk8s status | grep -q "helm3.*enabled"; then
        echo -e "${YELLOW}Habilitando Helm3 en MicroK8s...${NC}"
        microk8s enable helm3
    fi
    HELM="microk8s helm3"
else
    # Verificar que kubectl está instalado
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}Error: kubectl no está instalado${NC}"
        echo "Por favor, instale kubectl antes de continuar"
        exit 1
    fi

    # Verificar que hay un clúster Kubernetes disponible
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}Error: No se puede conectar al clúster de Kubernetes${NC}"
        echo "Por favor, asegúrese de que el clúster está en ejecución"
        exit 1
    fi

    # Verificar que Helm está instalado, si no, instalarlo
    if ! command -v helm &> /dev/null; then
        echo -e "${YELLOW}Helm no está instalado. Instalando Helm...${NC}"
        sudo snap install helm --classic || {
            # Si snap falla, intentar el método de script
            curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
            chmod 700 get_helm.sh
            ./get_helm.sh
            rm get_helm.sh
        }
        
        if ! command -v helm &> /dev/null; then
            echo -e "${RED}Error: No se pudo instalar Helm. Requiere instalación manual.${NC}"
            exit 1
        fi
        echo -e "${GREEN}Helm instalado correctamente.${NC}"
    fi
    HELM="helm"
fi

# Verificar si el namespace está siendo terminado
echo -e "${YELLOW}Verificando estado del namespace mlops...${NC}"
if $KUBECTL get namespace | grep -q mlops; then
    NAMESPACE_STATUS=$($KUBECTL get namespace mlops -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$NAMESPACE_STATUS" = "Terminating" ]; then
        echo -e "${RED}El namespace mlops está en estado de terminación. Forzando eliminación...${NC}"
        
        # Forzar eliminación del namespace usando el API directamente
        $KUBECTL get namespace mlops -o json | jq '.spec.finalizers = []' > tmp-ns.json
        $KUBECTL proxy &
        PROXY_PID=$!
        sleep 3
        curl -X PUT --data-binary @tmp-ns.json -H "Content-Type: application/json" http://127.0.0.1:8001/api/v1/namespaces/mlops/finalize
        kill $PROXY_PID
        rm tmp-ns.json
        
        # Esperar a que termine el namespace
        echo "Esperando a que el namespace anterior termine completamente..."
        for i in {1..30}; do
            if ! $KUBECTL get namespace | grep -q mlops; then
                break
            fi
            echo -n "."
            sleep 2
        done
        echo ""
        
        if $KUBECTL get namespace | grep -q mlops; then
            echo -e "${RED}No se pudo eliminar el namespace. Por favor, ejecute reset-kubernetes.sh y vuelva a intentarlo.${NC}"
            exit 1
        fi
    fi
fi

# Crear namespace si no existe
echo -e "${YELLOW}Creando namespace mlops...${NC}"
$KUBECTL create namespace mlops 2>/dev/null || echo "Namespace mlops ya existe"

# Crear directorios de datos en el host si no existen
echo -e "${YELLOW}Creando directorios de datos en el host...${NC}"
sudo mkdir -p /mnt/data/{postgres-mlflow,postgres-airflow,minio,airflow-dags}
sudo chmod -R 777 /mnt/data

# Desplegar MLflow
echo -e "${YELLOW}Desplegando MLflow y sus dependencias...${NC}"
$KUBECTL apply -f mlflow/postgres-minio.yaml
echo "Esperando 10 segundos para que PostgreSQL y MinIO se inicien..."
sleep 10
$KUBECTL apply -f mlflow/mlflow-deployment.yaml
$KUBECTL apply -f mlflow/mlflow-services.yaml

# Eliminar cualquier servicio de Airflow existente que pueda causar conflictos con Helm
echo -e "${YELLOW}Eliminando servicios de Airflow existentes...${NC}"
$KUBECTL delete service airflow-webserver -n mlops 2>/dev/null || true
$KUBECTL delete service postgres-airflow -n mlops 2>/dev/null || true

# Desplegar Airflow con Helm
echo -e "${YELLOW}Desplegando Airflow con Helm...${NC}"

# Añadir repositorio de Airflow si no existe
if ! $HELM repo list | grep -q "apache-airflow"; then
    echo "Añadiendo repositorio de Helm para Apache Airflow..."
    $HELM repo add apache-airflow https://airflow.apache.org
    $HELM repo update
fi

# Verificar la versión de chart disponible y adaptar el values.yaml
echo "Verificando versión del Helm chart de Airflow..."
AIRFLOW_CHART_VERSION=$($HELM search repo apache-airflow/airflow -o json | jq -r '.[0].version')
echo "Versión del chart de Airflow: $AIRFLOW_CHART_VERSION"

# Verificar y actualizar values.yaml si es necesario
echo "Comprobando compatibilidad del values.yaml con la versión del chart..."
$HELM show values apache-airflow/airflow > /tmp/airflow-reference-values.yaml

# Verificando si hay que actualizar el formato de config en values.yaml
if grep -q "config:" airflow/values.yaml; then
    CONFIG_FORMAT=$(grep -A 5 "config:" /tmp/airflow-reference-values.yaml)
    if [[ "$CONFIG_FORMAT" == *"type: object"* ]]; then
        echo -e "${YELLOW}Detectado formato de configuración incompatible. Actualizando values.yaml...${NC}"
        # Hacer una copia de seguridad del values.yaml original
        cp airflow/values.yaml airflow/values.yaml.bak
        
        # Actualizar el formato de configuración
        sed -i 's/config:/config:/g' airflow/values.yaml
        sed -i 's/  AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: "True"/  airflow:/g' airflow/values.yaml
        sed -i 's/  AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS: "False"/    config:/g' airflow/values.yaml
        sed -i 's/  AIRFLOW__WEBSERVER__EXPOSE_CONFIG: "True"/      AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: True/g' airflow/values.yaml
        sed -i 's/  AIRFLOW__CORE__LOAD_EXAMPLES: "False"/      AIRFLOW__CORE__LOAD_DEFAULT_CONNECTIONS: False\n      AIRFLOW__WEBSERVER__EXPOSE_CONFIG: True\n      AIRFLOW__CORE__LOAD_EXAMPLES: False/g' airflow/values.yaml
        
        # Corregir dags.gitSync
        if grep -q "dags.gitSync" airflow/values.yaml; then
            sed -i '/dest:/d' airflow/values.yaml
            sed -i '/syncWait:/d' airflow/values.yaml
        fi
        
        # Corregir webserver.service
        if grep -q "webserver.service" airflow/values.yaml; then
            sed -i '/nodePort:/d' airflow/values.yaml
        fi
        
        echo -e "${GREEN}values.yaml actualizado para compatibilidad con la versión $AIRFLOW_CHART_VERSION${NC}"
    fi
fi

# Aplicar volúmenes persistentes para Airflow
echo "Aplicando volúmenes persistentes para Airflow..."
$KUBECTL apply -f airflow/persistent-volumes.yaml

# Desinstalar Airflow si ya existe
if $HELM list -n mlops | grep -q "airflow"; then
    echo "Desinstalando versión anterior de Airflow..."
    $HELM uninstall airflow -n mlops
    # Dar tiempo para que se eliminen todos los recursos
    sleep 10
fi

# Instalar Airflow con Helm
echo "Instalando Airflow con Helm..."
$HELM install airflow apache-airflow/airflow --namespace mlops --values airflow/values.yaml

# Verificar que todos los pods están en ejecución
echo -e "${YELLOW}Esperando a que todos los pods estén listos...${NC}"
echo "Este proceso puede tardar varios minutos. Por favor, sea paciente."
# Esperar un poco antes de verificar el estado de los pods
sleep 30
$KUBECTL wait --for=condition=ready pod --all -n mlops --timeout=300s || echo -e "${RED}Algunos pods no están listos aún, verificar con '$KUBECTL get pods -n mlops'${NC}"

# Mostrar información de los servicios
echo -e "${GREEN}=== Servicios desplegados ===${NC}"
$KUBECTL get svc -n mlops

# Obtener IPs y puertos de los servicios
NODE_IP=$(hostname -I | awk '{print $1}')
MLFLOW_PORT=$($KUBECTL get svc mlflow -n mlops -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "pendiente")
AIRFLOW_PORT=$($KUBECTL get svc airflow-webserver -n mlops -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "pendiente")
MINIO_API_PORT=$($KUBECTL get svc minio -n mlops -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "pendiente")
MINIO_CONSOLE_PORT=$($KUBECTL get svc minio -n mlops -o jsonpath='{.spec.ports[1].nodePort}' 2>/dev/null || echo "pendiente")

echo -e "${GREEN}=== Acceso a los servicios ===${NC}"
echo -e "MLflow UI:\t http://$NODE_IP:$MLFLOW_PORT"
echo -e "Airflow UI:\t http://$NODE_IP:$AIRFLOW_PORT"
echo -e "MinIO API:\t http://$NODE_IP:$MINIO_API_PORT"
echo -e "MinIO Console:\t http://$NODE_IP:$MINIO_CONSOLE_PORT"
echo
echo -e "${YELLOW}Credenciales:${NC}"
echo -e "Airflow: usuario=airflow, contraseña=airflow"
echo -e "MinIO: usuario=minioadmin, contraseña=minioadmin"
echo
echo -e "${GREEN}Despliegue completado.${NC}"

echo -e "${YELLOW}=== Notas importantes ===${NC}"
echo -e "1. Si no puedes acceder a los servicios, verifica que los puertos NodePort están abiertos:"
echo -e "   sudo ufw allow $MLFLOW_PORT/tcp"
echo -e "   sudo ufw allow $AIRFLOW_PORT/tcp"
echo -e "   sudo ufw allow $MINIO_API_PORT/tcp"
echo -e "   sudo ufw allow $MINIO_CONSOLE_PORT/tcp"
echo
echo -e "2. Para ver los logs de los pods:"
echo -e "   $KUBECTL logs -n mlops -l app=mlflow"
echo -e "   $KUBECTL logs -n mlops -l component=webserver -c webserver # Para Airflow"
echo
echo -e "3. Para verificar el estado de la instalación de Airflow:"
echo -e "   $HELM list -n mlops"
echo -e "   $KUBECTL get pods -n mlops"
echo
echo -e "4. Para verificar los DAGs en el pod de Airflow:"
echo -e "   $KUBECTL exec -it -n mlops \$($KUBECTL get pods -n mlops -l component=scheduler -o name | head -1) -c scheduler -- ls -la /opt/airflow/dags/repo/"
echo
echo -e "5. Para eliminar todo el despliegue:"
echo -e "   $HELM uninstall airflow -n mlops"
echo -e "   $KUBECTL delete namespace mlops"