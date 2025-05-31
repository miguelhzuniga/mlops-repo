#!/bin/bash

echo "Eliminando despliegue completo de recursos en mlops-project..."

# Borrar los deployments en el namespace mlops-project
kubectl delete deployments --all -n mlops-project

# Borrar los services en el namespace mlops-project
kubectl delete services --all -n mlops-project

# Borrar los pods en el namespace mlops-project (por si hay pods huérfanos)
kubectl delete pods --all -n mlops-project

# Borrar los PersistentVolumeClaims si existen (eliminar almacenamiento persistente)
kubectl delete pvc --all -n mlops-project

# Borrar los jobs (como el job de inicialización de MinIO)
kubectl delete jobs --all -n mlops-project

# Borrar ConfigMaps si existen
kubectl delete configmaps --all -n mlops-project

# Borrar los secretos si hay alguno (usar con cuidado en entornos productivos)
kubectl delete secrets --all -n mlops-project

#Forzar
#!/bin/bash

NAMESPACE="mlops-project"

echo "Buscando pods en estado Terminating en namespace $NAMESPACE..."

pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}')
terminating_pods=$(kubectl get pods -n "$NAMESPACE" | grep Terminating | awk '{print $1}')

if [[ -z "$terminating_pods" ]]; then
  echo "No hay pods en estado Terminating."
else
  echo "Pods en Terminating encontrados:"
  echo "$terminating_pods"
  for pod in $terminating_pods; do
    echo "Forzando eliminación de pod $pod ..."
    kubectl delete pod "$pod" -n "$NAMESPACE" --grace-period=0 --force
  done
fi

echo "Buscando PVCs en namespace $NAMESPACE..."

pvcs=$(kubectl get pvc -n "$NAMESPACE" --field-selector=status.phase=Bound -o jsonpath='{.items[*].metadata.name}')
terminating_pvcs=$(kubectl get pvc -n "$NAMESPACE" | grep Terminating | awk '{print $1}')

if [[ -z "$terminating_pvcs" ]]; then
  echo "No hay PVCs en estado Terminating."
else
  echo "PVCs en Terminating encontrados:"
  echo "$terminating_pvcs"
  for pvc in $terminating_pvcs; do
    echo "Forzando eliminación de PVC $pvc ..."
    kubectl delete pvc "$pvc" -n "$NAMESPACE" --grace-period=0 --force
  done
fi

echo "Intentando eliminar namespace $NAMESPACE..."
kubectl delete namespace "$NAMESPACE" --ignore-not-found

echo "Proceso completado. Verifica el estado con: kubectl get namespace $NAMESPACE"
