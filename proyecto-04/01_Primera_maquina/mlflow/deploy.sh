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

# Verificar que MinIO esté realmente funcional
echo "Verificando que MinIO esté realmente funcional..."
for i in {1..30}; do
  if sudo microk8s kubectl exec -n mlops-project deployment/minio -- mc --version >/dev/null 2>&1; then
    echo "✅ MinIO está respondiendo correctamente"
    break
  else
    echo "⏳ MinIO aún no está listo, esperando... ($i/30)"
    sleep 10
  fi
  
  if [ $i -eq 30 ]; then
    echo "❌ MinIO no respondió después de 5 minutos"
    exit 1
  fi
done

# Espera adicional para asegurar que MinIO esté completamente inicializado
echo "Esperando 30 segundos adicionales para inicialización completa de MinIO..."
sleep 30

# Inicializar bucket
echo "Inicializando bucket MinIO para artefactos de MLflow..."
sudo microk8s kubectl apply -f manifests/init-job.yaml

# Esperar y verificar que el job se complete
echo "Esperando a que el job de inicialización se complete..."
sudo microk8s kubectl wait --for=condition=complete job/minio-init -n mlops-project --timeout=120s

# Verificar que el bucket se creó realmente
echo "Verificando que el bucket se creó correctamente..."
if sudo microk8s kubectl exec -n mlops-project deployment/minio -- ls /data | grep -q mlflow-artifacts; then
  echo "✅ Bucket mlflow-artifacts creado exitosamente"
else
  echo "⚠️  Bucket no detectado, creando manualmente..."
  sudo microk8s kubectl exec -n mlops-project deployment/minio -- mkdir -p /data/mlflow-artifacts
  echo "✅ Bucket creado manualmente"
fi

# Desplegar MLflow
echo "Desplegando MLflow..."
sudo microk8s kubectl apply -f manifests/mlflow.yaml

# Esperar a que MLflow esté listo
echo "Esperando a que MLflow esté listo..."
sudo microk8s kubectl wait --for=condition=ready pod -l app=mlflow -n mlops-project --timeout=180s

# Verificar que MLflow esté respondiendo
echo "Verificando que MLflow esté funcional..."
NODE_IP=$(microk8s kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
for i in {1..20}; do
  if curl -f http://$NODE_IP:30500/health >/dev/null 2>&1; then
    echo "✅ MLflow está respondiendo correctamente"
    break
  else
    echo "⏳ MLflow aún no está listo, esperando... ($i/20)"
    sleep 15
  fi
done

# Configurar Ingress
echo "Configurando acceso mediante Ingress..."
sudo microk8s kubectl apply -f manifests/ingress.yaml

# Mostrar información de acceso
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