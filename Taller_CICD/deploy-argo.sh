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
 
check_kubernetes_connection() {
    echo "Verificando conexión al clúster Kubernetes..."
    if ! sudo microk8s kubectl get nodes &> /dev/null; then
        echo_error "No se puede conectar al clúster de Kubernetes."
        return 1
    fi
    echo_success "Conexión al clúster establecida correctamente"
    return 0
}
 
deploy_argocd() {
    echo_step "Desplegando Argo CD"
    check_kubernetes_connection || return 1
    echo "Creando namespace para Argo CD..."
    sudo microk8s kubectl create namespace argocd 2>/dev/null || echo "El namespace ya existe"
    echo "Aplicando manifiesto de Argo CD..."
    sudo microk8s kubectl apply -n argocd -f argo-cd/install.yaml
    echo "Esperando a que los pods de Argo CD estén listos..."
    for i in {1..12}; do
        if sudo microk8s kubectl get pods -n argocd 2>/dev/null | grep -q "argocd-server" && \
           sudo microk8s kubectl get pods -n argocd | grep "argocd-server" | grep -q "Running"; then
            echo_success "Pods de Argo CD están listos"
            break
        fi
        if [ $i -eq 12 ]; then
            echo_warning "Tiempo de espera agotado."
        else
            echo "Esperando... ($i/12)"
            sleep 20
        fi
    done
    local max_attempts=10
    local attempt=1
    local password=""
    echo "Obteniendo la contraseña inicial de Argo CD..."
    while [ $attempt -le $max_attempts ]; do
        password=$(sudo microk8s kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" 2>/dev/null | base64 -d)
        if [ -n "$password" ]; then
            echo_success "Contraseña inicial de Argo CD: $password"
            break
        fi
        echo "Intento $attempt/$max_attempts..."
        sleep 10
        ((attempt++))
    done
    if [ -f "argo-cd/app.yaml" ]; then
        echo "Configurando la aplicación en Argo CD..."
        read -p "Ingresa la URL de tu repositorio Git: " GIT_REPO
        sed -i "s|https://github.com/YOUR_USERNAME/MLOPS_PUJ.git|${GIT_REPO}|g" argo-cd/app.yaml
        echo "Aplicando configuración de la aplicación en Argo CD..."
        sudo microk8s kubectl apply --validate=false -f argo-cd/app.yaml
    fi
    mkdir -p port_forward_logs
    sudo microk8s kubectl port-forward svc/argocd-server -n argocd 8080:443 > port_forward_logs/argocd.log 2>&1 &
    ARGOCD_PF_PID=$!
    sudo bash -c "echo $ARGOCD_PF_PID > port_forward_logs/argocd.pid"
    echo_success "Port-forwarding para Argo CD iniciado"
    echo_success "Puedes acceder a Argo CD en: https://localhost:8080"
    echo_success "Usuario: admin, Contraseña: $password"
    if [ ! -f "stop_port_forwards.sh" ]; then
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
    echo "No se encontró información de port-forwarding, por favor validar"
fi
EOF
        chmod +x stop_port_forwards.sh
    fi
    echo_step "Argo CD desplegado correctamente"
    echo_success "Para detener el port-forwarding, ejecuta: ./stop_port_forwards.sh"
    return 0
}
 
deploy_argocd