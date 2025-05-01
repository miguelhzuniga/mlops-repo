#!/bin/bash

# Verificar si ya existe el namespace mlops
if ! microk8s kubectl get namespace mlops &> /dev/null; then
  microk8s kubectl create namespace mlops
  echo "Namespace creado: mlops"
else
  echo "Namespace mlops ya existe, continuando..."
fi

# Lista de servicios a exponer con sus puertos correspondientes
declare -A services
services["airflow-webserver"]=30080
services["airflow-flower"]=30555
services["airflow-postgresql"]=30432
services["airflow-redis"]=30379

# Exponer servicios existentes usando NodePort mediante parches directos
echo "Exponiendo servicios con NodePort..."
for service in "${!services[@]}"; do
  if microk8s kubectl get service "$service" -n mlops &> /dev/null; then
    nodeport=${services[$service]}
    echo "Exponiendo $service en NodePort $nodeport..."
    
    # Usar patch para modificar el servicio directamente
    microk8s kubectl patch service "$service" -n mlops -p '{"spec": {"type": "NodePort"}}'
    
    # Obtener el primer puerto del servicio
    port=$(microk8s kubectl get service "$service" -n mlops -o jsonpath='{.spec.ports[0].port}')
    
    # Aplicar nodePort al primer puerto
    microk8s kubectl patch service "$service" -n mlops --type='json' -p="[{'op': 'replace', 'path': '/spec/ports/0/nodePort', 'value': $nodeport}]"
    
    echo "Servicio $service configurado para usar NodePort $nodeport"
  else
    echo "Servicio $service no encontrado, omitiendo..."
  fi
done

# Configurar GitSync para los DAGs desde tu repositorio
echo "Configurando GitSync para sincronizar DAGs desde tu repositorio GitHub..."

# Crear PVC para los DAGs si no existe
if ! microk8s kubectl get pvc dags-pvc -n mlops &> /dev/null; then
  echo "Creando PersistentVolumeClaim para DAGs..."
  cat <<EOF | microk8s kubectl apply -n mlops -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dags-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
EOF
fi

# Crear ConfigMap para GitSync si no existe
if ! microk8s kubectl get configmap git-sync-config -n mlops &> /dev/null; then
  echo "Creando ConfigMap para GitSync..."
  cat <<EOF | microk8s kubectl apply -n mlops -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: git-sync-config
data:
  GIT_SYNC_REPO: "https://github.com/miguelhzuniga/mlops-repo.git"
  GIT_SYNC_BRANCH: "luis/proyecto-03"
  GIT_SYNC_DEPTH: "1"
  GIT_SYNC_WAIT: "60"
  GIT_SYNC_ROOT: "/git"
  GIT_SYNC_DEST: "repo"
  GIT_SYNC_SUBPATH: "AIRFLOW/dags"
EOF
fi

# Crear o actualizar el Deployment de GitSync
echo "Creando/actualizando deployment de GitSync..."
cat <<EOF | microk8s kubectl apply -n mlops -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: git-sync
  labels:
    app: git-sync
spec:
  replicas: 1
  selector:
    matchLabels:
      app: git-sync
  template:
    metadata:
      labels:
        app: git-sync
    spec:
      containers:
      - name: git-sync
        image: k8s.gcr.io/git-sync:v3.1.6
        envFrom:
        - configMapRef:
            name: git-sync-config
        volumeMounts:
        - name: dags-volume
          mountPath: /git
      volumes:
      - name: dags-volume
        persistentVolumeClaim:
          claimName: dags-pvc
EOF

# Verificar que los pods estén listos
echo "Esperando a que los pods estén listos..."
microk8s kubectl rollout status deployment git-sync -n mlops --timeout=60s || true

# Mostrar servicios
echo "Servicios expuestos via NodePort:"
microk8s kubectl get services -n mlops

echo "Ahora puedes acceder al webserver de Airflow en:"
echo "  - Desde este servidor: http://localhost:30080"
echo "  - Desde otra máquina: http://$(hostname -I | awk '{print $1}'):30080"
echo "Tu repositorio Git se sincronizará automáticamente con la carpeta de DAGs"
echo "¡Despliegue completado!"