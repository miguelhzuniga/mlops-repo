#!/bin/bash

echo "Eliminando Airflow del cluster mlops-project..."

echo "Desinstalando Airflow..."
sudo microk8s helm3 uninstall airflow -n mlops-project

echo "Forzando eliminación de pods de Airflow..."
sudo microk8s kubectl delete pods -l app.kubernetes.io/name=airflow -n mlops-project --force --grace-period=0

echo "Eliminando servicios de Airflow..."
sudo microk8s kubectl delete services -l app.kubernetes.io/name=airflow -n mlops-project
sudo microk8s kubectl delete deployments -l app.kubernetes.io/name=airflow -n mlops-project
sudo microk8s kubectl delete statefulsets -l app.kubernetes.io/name=airflow -n mlops-project

echo "Eliminando almacenamiento persistente de Airflow..."
sudo microk8s kubectl delete pvc -l app.kubernetes.io/name=airflow -n mlops-project

echo "Eliminando secrets y configmaps de Airflow..."
sudo microk8s kubectl delete secrets -l app.kubernetes.io/name=airflow -n mlops-project
sudo microk8s kubectl delete configmaps -l app.kubernetes.io/name=airflow -n mlops-project

echo "Eliminando jobs de Airflow..."
sudo microk8s kubectl delete jobs -l app.kubernetes.io/name=airflow -n mlops-project

echo "Esperando limpieza completa..."
sleep 15

echo "Verificando limpieza..."
sudo microk8s kubectl get all -n mlops-project

echo "Airflow eliminado completamente!"