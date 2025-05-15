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
    
    # Verificar si el archivo de datos existe
    if [ ! -f "data/iris.csv" ]; then
        echo "Generando datos de muestra..."
        # Crear un archivo Python temporal
        cat > generate_data.py << EOF
from sklearn.datasets import load_iris
import pandas as pd
import os

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
os.makedirs('data', exist_ok=True)
df.to_csv('data/iris.csv', index=False)
print('Datos de ejemplo generados en data/iris.csv')
EOF
        # Ejecutar el script Python
        python3 generate_data.py
        # Eliminar el archivo temporal
        rm generate_data.py
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
    
    echo "Aplicando manifiestos..."
    microk8s kubectl apply -k .
    
    cd ..
    
    echo_success "Aplicación desplegada correctamente"
}

setup_argocd() {
    echo_step "Instalando y configurando Argo CD"
    
    echo "Creando namespace para Argo CD..."
    microk8s kubectl create namespace argocd 2>/dev/null || echo "El namespace ya existe"
    
    echo "Instalando Argo CD..."
    microk8s kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
    
    echo "Esperando a que los pods de Argo CD estén listos..."
    microk8s kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=argocd-server -n argocd --timeout=300s || echo "Tiempo de espera agotado, continuando..."
    
    echo "Configurando la aplicación en Argo CD..."
    
    # Pregunta por el repositorio Git
    read -p "Ingresa la URL de tu repositorio Git (ej. https://github.com/tu-usuario/tu-repo.git): " GIT_REPO
    
    # Actualizar la URL del repositorio en el archivo de Argo CD
    if [ -f "argo-cd/app.yaml" ]; then
        sed -i "s|https://github.com/YOUR_USERNAME/MLOPS_PUJ.git|${GIT_REPO}|g" argo-cd/app.yaml
        microk8s kubectl apply -f argo-cd/app.yaml
        echo_success "Aplicación configurada en Argo CD"
    else
        echo_warning "El archivo argo-cd/app.yaml no existe. No se pudo configurar la aplicación en Argo CD automáticamente."
    fi
    
    # Obtener la contraseña inicial de Argo CD
    echo -n "Contraseña inicial de Argo CD: "
    ARGOCD_PASSWORD=$(microk8s kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
    echo "${ARGOCD_PASSWORD}"
    
    echo_success "Argo CD instalado y configurado correctamente"
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
    echo_warning "microk8s kubectl port-forward svc/argocd-server -n argocd 8080:443"
    
    echo_success "Ahora puedes acceder a:"
    echo_success "- API: http://localhost:8000 (ejecuta el comando port-forward para la API)"
    echo_success "- Prometheus: http://localhost:9090 (ejecuta el comando port-forward para Prometheus)"
    echo_success "- Grafana: http://localhost:3000 (ejecuta el comando port-forward para Grafana)"
    echo_success "  Usuario: admin, contraseña: admin"
    echo_success "- Argo CD: https://localhost:8080 (ejecuta el comando port-forward para Argo CD)"
    echo_success "  Usuario: admin, contraseña: la mostrada anteriormente"
}

main() {
    echo_step "Iniciando proceso de despliegue de MLOps"
    
    check_dependencies
    train_model
    build_docker_images
    start_microk8s
    deploy_to_kubernetes
    setup_argocd
    setup_port_forwarding
    
    echo_step "Proceso completado con éxito"
    echo_success "La aplicación ha sido desplegada correctamente en MicroK8s."
    echo_success "Para verificar el estado de los pods, ejecuta: microk8s kubectl get pods -n mlops-puj"
    echo_success "Para verificar el estado de Argo CD, ejecuta: microk8s kubectl get pods -n argocd"
}

main