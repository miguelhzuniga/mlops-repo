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
 
   
    if ! command -v kubectl &> /dev/null; then
        echo_warning "kubectl no está instalado. Instalando..."
       
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
    else
        echo_success "kubectl está instalado"
    fi
 
   
    if ! command -v microk8s &> /dev/null; then
        echo_warning "microk8s no está instalado. Instalando..."
       
        sudo snap install microk8s --classic
        sudo usermod -aG microk8s $USER
        echo_warning "Para que los cambios en el grupo microk8s surtan efecto, es posible que debas cerrar sesión y volver a iniciarla."
        echo_warning "O ejecuta: newgrp microk8s"
        mkdir -p ~/.kube
        sudo microk8s.kubectl config view --raw > ~/.kube/config
        sudo chown -R $USER ~/.kube
    else
        echo_success "microk8s está instalado"
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
   
   
    if [ ! -f "data/iris.csv" ]; then
        echo "Generando datos de muestra..."
        python3 -c "
from sklearn.datasets import load_iris
import pandas as pd
import os
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
os.makedirs('data', exist_ok=True)
df.to_csv('data/iris.csv', index=False)
print('Datos de ejemplo generados en data/iris.csv')
"
    fi
 
   
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
    echo_step "Iniciando microk8s"
   
   
    if microk8s status | grep -q "running"; then
        echo_warning "microk8s ya está en ejecución"
    else
        echo "Iniciando microk8s..."
        sudo microk8s start
    fi
   
    echo "Habilitando addons necesarios..."
    microk8s enable dns storage registry dashboard
   
    echo "Configurando acceso al registro de Docker de microk8s..."
    export DOCKER_REGISTRY="localhost:32000"
   
    echo_success "microk8s iniciado correctamente"
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
   
   
    echo "Aplicando manifiestos..."
    microk8s kubectl apply -k .
   
   
    cd ..
   
    echo_success "Aplicación desplegada correctamente"
}
 
setup_port_forwarding() {
    echo_step "Configurando port-forwarding para acceso a los servicios"
   
   
    echo "Esperando a que los pods estén listos..."
    microk8s kubectl wait --for=condition=ready pod -l app=ml-api -n mlops-puj --timeout=300s
    microk8s kubectl wait --for=condition=ready pod -l app=prometheus -n mlops-puj --timeout=300s
    microk8s kubectl wait --for=condition=ready pod -l app=grafana -n mlops-puj --timeout=300s
   
   
    echo "Exponiendo servicios en puertos locales..."
    echo_warning "Abre nuevas terminales para ejecutar estos comandos:"
    echo_warning "microk8s kubectl port-forward svc/ml-api-service 8000:8000 -n mlops-puj"
    echo_warning "microk8s kubectl port-forward svc/prometheus-service 9090:9090 -n mlops-puj"
    echo_warning "microk8s kubectl port-forward svc/grafana-service 3000:3000 -n mlops-puj"
   

    echo_success "Ahora puedes acceder a:"
    echo_success "- API: http://localhost:32675"
    echo_success "- Prometheus: http://localhost:30623"
    echo_success "- Grafana: http://localhost:32618 (usuario: admin, contraseña: admin)"
}
 
 
main() {
    echo_step "Iniciando proceso de despliegue de MLOps"
   
    check_dependencies
    train_model
    build_docker_images
    start_microk8s
    build_docker_images  
    deploy_to_kubernetes
    setup_port_forwarding
   
    echo_step "Proceso completado con éxito"
    echo_success "La aplicación ha sido desplegada correctamente en microk8s."
    echo_success "Para verificar el estado de los pods, ejecuta: microk8s kubectl get pods -n mlops-puj"
}
 
main