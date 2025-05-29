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

# Borrar el namespace si ya no lo necesitas
kubectl delete namespace mlops-project

# Esperar a que todos los recursos sean eliminados
echo "Esperando a que los recursos se eliminen..."
sleep 10

echo "Despliegue eliminado completamente."
