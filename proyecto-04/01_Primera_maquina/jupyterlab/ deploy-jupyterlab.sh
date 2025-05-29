#!/bin/bash
# deploy-jupyterlab.sh - Desplegar JupyterLab para experimentación MLOps

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="mlops-project"

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_cluster() {
    print_info "Verificando cluster existente..."
    
    if ! command -v microk8s >/dev/null 2>&1; then
        print_error "MicroK8s no está instalado."
        exit 1
    fi
    
    if ! command -v kubectl >/dev/null 2>&1; then
        sudo snap alias microk8s.kubectl kubectl
    fi
    
    if ! microk8s kubectl get namespace $NAMESPACE >/dev/null 2>&1; then
        print_error "Namespace $NAMESPACE no existe. Despliegue MLflow y Airflow primero."
        exit 1
    fi
    
    print_success "Cluster verificado"
}

deploy_jupyterlab() {
    print_info "Desplegando JupyterLab..."
    
    microk8s kubectl apply -f jupyterlab.yaml
    
    print_info "Esperando a que JupyterLab esté listo..."
    microk8s kubectl wait --for=condition=ready pod -l app=jupyterlab -n $NAMESPACE --timeout=180s || \
    print_warning "JupyterLab puede estar iniciando aún"
    
    print_success "JupyterLab desplegado"
}

show_info() {
    NODE_IP=$(microk8s kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
    
    print_success "¡JupyterLab desplegado exitosamente!"
    echo ""
    echo "=============================================="
    echo "ACCESO A JUPYTERLAB:"
    echo "=============================================="
    echo "🔬 JupyterLab: http://$NODE_IP:30888"
    echo "   Token: mlops123"
    echo "   Imagen: camilosvel/jupyterlab-mlops:latest"
    echo ""
    echo "CARACTERÍSTICAS:"
    echo "✅ Librerías MLOps preinstaladas"
    echo "✅ Variables de entorno configuradas para MLflow"
    echo "✅ Notebooks persistentes en /home/jovyan/work"
    echo "✅ Conexión automática a MLflow y MinIO"
    echo "=============================================="
    echo ""
    echo "EJEMPLO DE USO EN JUPYTER:"
    echo "import os"
    echo "import mlflow"
    echo "mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))"
    echo "=============================================="
}

main() {
    print_info " Desplegando JupyterLab para experimentación..."
    
    check_cluster
    deploy_jupyterlab
    show_info
    
    print_success " JupyterLab listo para experimentar!"
}

main "$@"