
#!/bin/bash
# join-node.sh - Script para unir un nodo al clúster MicroK8s y configurarlo para servicios específicos
 
# Verificar si microk8s está instalado y listo
if ! command -v microk8s >/dev/null 2>&1; then
  echo "MicroK8s no está instalado. Instalando..."
  sudo snap install microk8s --classic
  sudo usermod -a -G microk8s $USER
  sudo chown -f -R $USER ~/.kube
  echo "Iniciando nuevo grupo sin cerrar sesión..."
  newgrp microk8s
fi
 
# Verificar status de MicroK8s
echo "Verificando el estado de MicroK8s..."
microk8s status --wait-ready
 
# Crear alias para kubectl si no existe
if ! command -v kubectl >/dev/null 2>&1; then
  echo "Creando alias para kubectl..."
  sudo snap alias microk8s.kubectl kubectl
fi
 
# Solicitar el comando de unión
echo "=============================================="
echo "Ingresa el comando de unión generado por el nodo principal (microk8s join ...):"
read -p "> " JOIN_COMMAND
 
# Unirse al clúster
echo "Uniendo este nodo al clúster..."
$JOIN_COMMAND
 
# Esperar a que se una correctamente
echo "Esperando a que el nodo se una correctamente al clúster..."
sleep 20
 
# Obtener el nombre del nodo actual
NODE_NAME=$(hostname)
echo "Nombre del nodo: $NODE_NAME"
 
# Etiquetar el nodo con un valor único para poder identificarlo
echo "Etiquetando el nodo para permitir despliegues específicos en él..."
microk8s kubectl label node $NODE_NAME node-type=worker node-id=$NODE_NAME
 
# Crear un ejemplo de manifiesto para desplegar un servicio en este nodo específico
echo "Creando plantilla de ejemplo para desplegar servicios en este nodo específicamente..."
mkdir -p manifests-local
 
cat > manifests-local/node-specific-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: example-service-${NODE_NAME}
  namespace: mlops-project
spec:
  replicas: 1
  selector:
    matchLabels:
      app: example-service-${NODE_NAME}
  template:
    metadata:
      labels:
        app: example-service-${NODE_NAME}
    spec:
      nodeSelector:
        node-id: ${NODE_NAME}
      containers:
      - name: nginx
        image: nginx
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: example-service-${NODE_NAME}
  namespace: mlops-project
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 80
    nodePort: 30080
  selector:
    app: example-service-${NODE_NAME}
EOF
 
echo "=============================================="
echo "¡Nodo unido al clúster exitosamente y configurado para despliegues específicos!"
echo "Para verificar el estado del clúster desde el nodo principal:"
echo "  microk8s kubectl get nodes"
echo ""
echo "Para desplegar un servicio específicamente en este nodo:"
echo "  microk8s kubectl apply -f manifests-local/node-specific-deployment.yaml"
echo ""
echo "Puedes modificar el archivo de ejemplo para adaptarlo a tus necesidades"
echo "La clave está en usar el selector de nodo: nodeSelector: node-id: ${NODE_NAME}"
echo "=============================================="