#!/bin/bash

echo "Construyendo imagen personalizada de Airflow..."
cd docker
if sudo docker build -t airflow-custom:latest .; then
    echo "Imagen construida exitosamente"
else
    echo "Error construyendo imagen"
    exit 1
fi
cd ..

echo "Importando imagen a microk8s..."
sudo docker save airflow-custom:latest -o /tmp/airflow-custom.tar
sudo microk8s ctr images import /tmp/airflow-custom.tar
sudo rm /tmp/airflow-custom.tar

echo "Instalando Airflow en cluster mlops-project..."

echo "Agregando repositorio Helm..."
sudo microk8s helm3 repo add apache-airflow https://airflow.apache.org
sudo microk8s helm3 repo update

echo "Instalando Airflow..."
sudo microk8s helm3 install airflow apache-airflow/airflow \
  -n mlops-project \
  -f values.yaml

echo "Reiniciando pods para usar nueva imagen..."
sleep 10
sudo microk8s kubectl delete pods -n mlops-project -l tier=airflow

echo "Esperando que Airflow este listo..."
sudo microk8s kubectl wait --for=condition=ready pod -l component=scheduler -n mlops-project --timeout=300s

echo "Airflow instalado correctamente!"
echo "Accede en: http://localhost:30080"
echo "Verificar pods: sudo microk8s kubectl get pods -n mlops-project"