#!/bin/bash
# deploy-airflow.sh - Desplegar Airflow con PostgreSQL y pgAdmin

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="mlops-project"
AIRFLOW_RELEASE="airflow"

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
        print_error "Namespace $NAMESPACE no existe. Despliegue MLflow primero."
        exit 1
    fi
    
    print_success "Cluster verificado"
}

setup_helm() {
    print_info "Configurando Helm..."
    
    if ! microk8s helm3 version >/dev/null 2>&1; then
        microk8s enable helm3
        sleep 10
    fi
    
    microk8s helm3 repo add apache-airflow https://airflow.apache.org
    microk8s helm3 repo update
    
    print_success "Helm configurado"
}

deploy_airflow() {
    print_info "Desplegando Airflow con Helm..."
    
    if microk8s helm3 list -n $NAMESPACE | grep -q $AIRFLOW_RELEASE; then
        print_warning "Actualizando Airflow existente..."
        microk8s helm3 upgrade $AIRFLOW_RELEASE apache-airflow/airflow \
            --namespace $NAMESPACE \
            --values airflow-values.yaml \
            --wait --timeout=15m
    else
        microk8s helm3 install $AIRFLOW_RELEASE apache-airflow/airflow \
            --namespace $NAMESPACE \
            --values airflow-values.yaml \
            --wait --timeout=15m
    fi
    
    print_success "Airflow desplegado con Helm"
}

deploy_pgadmin() {
    print_info "Desplegando pgAdmin..."
    
    microk8s kubectl apply -f pgadmin.yaml
    
    print_info "Esperando a que pgAdmin esté listo..."
    microk8s kubectl wait --for=condition=ready pod -l app=pgadmin -n $NAMESPACE --timeout=120s || \
    print_warning "pgAdmin puede estar iniciando aún"
    
    print_success "pgAdmin desplegado"
}

setup_connections() {
    print_info "Configurando conexiones..."
    
    microk8s kubectl wait --for=condition=ready pod -l component=scheduler -n $NAMESPACE --timeout=300s
    
    microk8s kubectl exec -n $NAMESPACE deployment/airflow-scheduler -- \
        airflow connections add 'postgres_default' \
        --conn-type 'postgres' \
        --conn-host 'postgres.mlops-project.svc.cluster.local' \
        --conn-schema 'mlflow' \
        --conn-login 'mlflow' \
        --conn-password 'mlflow123' \
        --conn-port 5432 || print_warning "Conexión ya existe"
    
    print_success "Conexiones configuradas"
}

show_info() {
    NODE_IP=$(microk8s kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
    
    print_success "¡Airflow desplegado con Helm!"
    echo ""
    echo "=============================================="
    echo "ACCESO A SERVICIOS:"
    echo "=============================================="
    echo "🌐 Airflow Web UI: http://$NODE_IP:32080"
    echo "   Usuario: airflow / Contraseña: airflow"
    echo ""
    echo "🐘 pgAdmin: http://$NODE_IP:30050"
    echo "   Usuario: admin@example.com / Contraseña: admin"
    echo ""
    echo "=============================================="
    echo "PRÓXIMO PASO:"
    echo "=============================================="
    echo "Para desplegar JupyterLab:"
    echo "  cd ../jupyterlab && ./deploy-jupyterlab.sh"
    echo "=============================================="
}

main() {
    print_info " Desplegando Airflow con Helm..."
    
    check_cluster
    setup_helm
    deploy_airflow
    deploy_pgadmin
    setup_connections
    show_info
    
    print_success " Airflow desplegado!"
}

main "$@"