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

delete_namespace_if_exists() {
    local ns="$1"
    echo_step "Eliminando recursos del namespace '$ns'"

    if sudo kubectl get namespace "$ns" &>/dev/null; then
        sudo kubectl delete all --all -n "$ns"
        sudo kubectl delete namespace "$ns"
        echo_success "Namespace '$ns' eliminado"
    else
        echo_warning "El namespace '$ns' no existe"
    fi
}

stop_minikube() {
    echo_step "Deteniendo minikube"
    if minikube status | grep -q "Running"; then
        minikube stop
        echo_success "minikube detenido"
    else
        echo_warning "minikube no está corriendo"
    fi
}

remove_local_docker_images() {
    echo_step "Eliminando imágenes Docker locales (opcional)"

    DOCKER_USERNAME="${DOCKER_USERNAME:-mlops-puj}"

    docker rmi -f ${DOCKER_USERNAME}/ml-api:latest || echo_warning "Imagen ml-api no encontrada"
    docker rmi -f ${DOCKER_USERNAME}/load-tester:latest || echo_warning "Imagen load-tester no encontrada"

    echo_success "Imágenes Docker eliminadas (si existían)"
}

main() {
    echo_step "Iniciando desmontaje del entorno de MLOps"

    delete_namespace_if_exists "mlops-puj"
    delete_namespace_if_exists "argocd"
    stop_minikube
    remove_local_docker_images

    echo_success "Entorno desmontado completamente"
}

main
