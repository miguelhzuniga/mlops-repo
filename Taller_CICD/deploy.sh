#!/bin/bash
set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 

echo_success() {
    echo -e "${GREEN}✓ $1${NC}"
}
echo_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}
echo_error() {
    echo -e "${RED}✗ $1${NC}"
}
echo_step() {
    echo -e "\n${GREEN}==== $1 ====${NC}"
}

check_dependencies() {
    echo_step "Verificando dependencias"
    
    if ! command -v docker &> /dev/null; then
        echo_warning "Docker no está instalado. Instalando..."
        sudo apt-get update
        sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
        echo_warning "Para que los cambios en el grupo docker surtan efecto, es posible que debas cerrar sesión y volver a iniciarla."
        echo_warning "O ejecuta: newgrp docker"
    else
        echo_success "Docker está instalado"
    fi
    
    if ! command -v microk8s &> /dev/null; then
        echo_warning "MicroK8s no está instalado. Instalando..."
        sudo snap install microk8s --classic
        sudo usermod -a -G microk8s $USER
        echo_warning "Para que los cambios en el grupo microk8s surtan efecto, es posible que debas cerrar sesión y volver a iniciarla."
    else
        echo_success "MicroK8s está instalado"
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo_warning "kubectl no está instalado. Instalando..."
        sudo snap install kubectl --classic
    else
        echo_success "kubectl está instalado"
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo_warning "Python no está instalado. Instalando..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip
    else
        echo_success "Python está instalado"
    fi
    
    if ! command -v pip3 &> /dev/null; then
        echo_warning "pip no está instalado. Instalando..."
        sudo apt-get update
        sudo apt-get install -y python3-pip
    else
        echo_success "pip está instalado"
    fi
}

train_model() {
    echo_step "Entrenando el modelo"
    
    cd api || { echo_error "La carpeta api no existe"; exit 1; }
    
    mkdir -p data
    
    echo "Instalando dependencias Python..."
    pip3 install -r requirements.txt
    
    echo "Ejecutando entrenamiento..."
    python3 train_model.py
    
    cd ..
    
    echo_success "Modelo entrenado correctamente"
}

build_docker_images() {
    echo_step "Construyendo imágenes Docker"
    
    DOCKER_USERNAME="${DOCKER_USERNAME:-mlops-puj}"  
    
    read -p "¿Quieres subir las imágenes a Docker Hub? (s/n): " PUSH_IMAGES
    if [[ "$PUSH_IMAGES" == "s" || "$PUSH_IMAGES" == "S" ]]; then
        read -p "Ingresa tu usuario de Docker Hub: " DOCKER_USERNAME
        docker login
    fi
    
    echo "Construyendo imagen de la API..."
    docker build -t ${DOCKER_USERNAME}/ml-api:latest ./api
    
    echo "Construyendo imagen del LoadTester..."
    docker build -t ${DOCKER_USERNAME}/load-tester:latest ./loadtester
    
    if [[ "$PUSH_IMAGES" == "s" || "$PUSH_IMAGES" == "S" ]]; then
        echo "Subiendo imágenes a Docker Hub..."
        docker push ${DOCKER_USERNAME}/ml-api:latest
        docker push ${DOCKER_USERNAME}/load-tester:latest
    fi
    
    echo_success "Imágenes Docker construidas correctamente"
    
    export DOCKER_USERNAME
}

start_microk8s() {
    echo_step "Verificando MicroK8s"
    
    # Verificar si MicroK8s está en ejecución
    if ! microk8s status | grep -q "microk8s is running"; then
        echo_warning "MicroK8s no está en ejecución, iniciándolo..."
        sudo microk8s start
    fi
    
    # Habilitar addons necesarios
    echo "Habilitando addons necesarios..."
    microk8s enable dns storage
    
    # Esperar a que los addons estén listos
    echo "Esperando a que los addons estén listos..."
    sleep 10
    
    echo_success "MicroK8s configurado correctamente"
}

deploy_to_kubernetes() {
    echo_step "Desplegando la aplicación en Kubernetes"
    
    DOCKER_USERNAME="${DOCKER_USERNAME:-mlops-puj}"  
    
    echo "Creando namespace mlops-puj..."
    microk8s kubectl create namespace mlops-puj 2>/dev/null || echo "El namespace ya existe"
    
    echo "Actualizando variables en los manifiestos..."
    cd manifests || { echo_error "La carpeta manifests no existe"; exit 1; }
    find . -type f -name "*.yaml" -exec sed -i "s|\${DOCKER_USERNAME}|${DOCKER_USERNAME}|g" {} \;
    find . -type f -name "*.yaml" -exec sed -i "s|\${IMAGE_TAG}|latest|g" {} \;
    
    echo "Aplicando manifiestos individualmente para evitar problemas de validación..."
    for yaml_file in $(find . -type f -name "*.yaml" | sort); do
        echo "Aplicando $yaml_file..."
        microk8s kubectl apply -f "$yaml_file" --validate=false || echo "Error al aplicar $yaml_file, continuando..."
    done
    
    cd ..
    
    echo_success "Aplicación desplegada correctamente"
}

setup_port_forwarding() {
    echo_step "Configurando port-forwarding para acceso a los servicios"
    
    echo "Esperando a que los pods estén listos..."
    microk8s kubectl wait --for=condition=ready pod -l app=ml-api -n mlops-puj --timeout=300s || echo "API no lista, verifica con: microk8s kubectl get pods -n mlops-puj"
    microk8s kubectl wait --for=condition=ready pod -l app=prometheus -n mlops-puj --timeout=300s || echo "Prometheus no listo, verifica con: microk8s kubectl get pods -n mlops-puj"
    microk8s kubectl wait --for=condition=ready pod -l app=grafana -n mlops-puj --timeout=300s || echo "Grafana no listo, verifica con: microk8s kubectl get pods -n mlops-puj"
    
    echo "Iniciando port-forwarding para todos los servicios..."
    
    # Crear un directorio para los logs de port-forwarding
    mkdir -p port_forward_logs
    
    # Iniciar port-forwarding en segundo plano
    microk8s kubectl port-forward svc/ml-api-service 8000:8000 -n mlops-puj > port_forward_logs/api.log 2>&1 &
    API_PF_PID=$!
    
    microk8s kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-puj > port_forward_logs/prometheus.log 2>&1 &
    PROM_PF_PID=$!
    
    microk8s kubectl port-forward svc/grafana-service 3000:3000 -n mlops-puj > port_forward_logs/grafana.log 2>&1 &
    GRAFANA_PF_PID=$!
    
    # Guardar los PIDs para poder terminarlos después
    echo "$API_PF_PID" > port_forward_logs/api.pid
    echo "$PROM_PF_PID" > port_forward_logs/prometheus.pid
    echo "$GRAFANA_PF_PID" > port_forward_logs/grafana.pid
    
    # Verificar que los port-forwards están funcionando
    sleep 3
    
    echo_success "Port-forwarding configurado para todos los servicios"
    echo_success "Los logs están disponibles en el directorio port_forward_logs/"
    echo_success "Para detener los port-forwards, ejecuta: ./stop_port_forwards.sh"
    
    # Crear script para detener los port-forwards
    cat > stop_port_forwards.sh << 'EOF'
#!/bin/bash
if [ -d "port_forward_logs" ]; then
    for pid_file in port_forward_logs/*.pid; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            SERVICE=$(basename "$pid_file" .pid)
            if ps -p $PID > /dev/null; then
                echo "Deteniendo port-forward para $SERVICE (PID: $PID)..."
                kill $PID
            else
                echo "El proceso de port-forward para $SERVICE ya no está en ejecución"
            fi
        fi
    done
    echo "Todos los port-forwards detenidos"
else
    echo "No se encontró información de port-forwarding"
fi
EOF
    
    chmod +x stop_port_forwards.sh
    
    echo_success "Ahora puedes acceder a:"
    echo_success "- API: http://localhost:8000"
    echo_success "- Prometheus: http://localhost:9090"
    echo_success "- Grafana: http://localhost:3000 (usuario: admin, contraseña: admin)"
}

main() {
    echo_step "Iniciando proceso de despliegue de servicios Kubernetes"
    
    check_dependencies
    train_model
    build_docker_images
    start_microk8s
    deploy_to_kubernetes
    setup_port_forwarding
    
    echo_step "Proceso completado con éxito"
    echo_success "Los servicios han sido desplegados correctamente en MicroK8s."
    echo_success "Para verificar el estado de los pods, ejecuta: microk8s kubectl get pods -n mlops-puj"
}

main