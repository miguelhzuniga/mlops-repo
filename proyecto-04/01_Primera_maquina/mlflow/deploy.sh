#!/bin/bash

# Verificar si microk8s está instalado y listo
if ! command -v microk8s >/dev/null 2>&1; then
  echo "MicroK8s no está instalado. Instalando..."
  sudo snap install microk8s --classic
  sudo usermod -a -G microk8s $USER
  echo "Por favor, cierre la sesión y vuelva a iniciarla para que los cambios surtan efecto."
  exit 1
fi

# Habilitar addons necesarios
echo "Habilitando addons necesarios..."
sudo microk8s enable dns storage helm3 ingress dashboard registry

# Esperar a que los servicios estén listos
echo "Esperando a que los servicios estén listos..."
sleep 10

# Crear alias para kubectl si no existe
if ! command -v kubectl >/dev/null 2>&1; then
  echo "Creando alias para kubectl..."
  sudo snap alias microk8s.kubectl kubectl
fi

# Crear namespace
echo "Creando namespace..."
sudo microk8s kubectl apply -f manifests/namespace.yaml

# Desplegar almacenamiento
echo "Desplegando servicios de almacenamiento (PostgreSQL y MinIO)..."
sudo microk8s kubectl apply -f manifests/storage.yaml

# Esperar a que los pods estén listos
echo "Esperando a que los pods de almacenamiento estén listos..."
sudo microk8s kubectl wait --for=condition=ready pod -l app=postgres -n mlops-project --timeout=120s
sudo microk8s kubectl wait --for=condition=ready pod -l app=minio -n mlops-project --timeout=120s

# Inicializar bucket
echo "Inicializando bucket MinIO para artefactos de MLflow..."
sudo microk8s kubectl apply -f manifests/init-job.yaml
sleep 5
sudo microk8s kubectl wait --for=condition=complete job/minio-init -n mlops-project --timeout=60s

# Desplegar MLflow
echo "Desplegando MLflow..."
sudo microk8s kubectl apply -f manifests/mlflow.yaml

# Esperar a que MLflow esté listo
echo "Esperando a que MLflow esté listo..."
sudo microk8s kubectl wait --for=condition=ready pod -l app=mlflow -n mlops-project --timeout=120s

# Configurar Ingress
echo "Configurando acceso mediante Ingress..."
sudo microk8s kubectl apply -f manifests/ingress.yaml

# Mostrar información de acceso
NODE_IP=$(microk8s kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
echo ""
echo "¡Despliegue completado exitosamente!"
echo "=============================================="
echo "Accede a MLflow en: http://$NODE_IP:30500"
echo "Accede a la consola de MinIO en: http://$NODE_IP:30901"
echo "Credenciales de MinIO:"
echo "  Usuario: adminuser"
echo "  Contraseña: securepassword123"
echo "=============================================="
echo "Para verificar el estado de los pods:"
echo "  microk8s kubectl get pods -n mlops-project"
echo ""
echo " Argo CD se encargará de las actualizaciones automáticas"
echo " Para cambios: solo haz push al repositorio"