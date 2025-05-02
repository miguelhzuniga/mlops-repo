#!/bin/bash

echo "Eliminando despliegue de MLflow..."

kubectl delete -f manifests/ingress.yaml
kubectl delete -f manifests/mlflow.yaml
kubectl delete -f manifests/init-job.yaml
kubectl delete -f manifests/storage.yaml

echo "Esperando a que los recursos se eliminen..."
sleep 10

kubectl delete -f manifests/namespace.yaml

echo "Despliegue eliminado completamente."