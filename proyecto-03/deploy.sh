#!/bin/bash
sudo mkdir -p /mnt/airflow/postgres-data
sudo chmod -R 777 /mnt/airflow

if ! microk8s kubectl get namespace mlops &> /dev/null; then
  microk8s kubectl create namespace mlops
  echo "Created namespace: mlops"
else
  echo "Namespace mlops already exists, continuing..."
fi

DOCKER_COMPOSE_PATH="./AIRFLOW/docker-compose.yml"
DAG_PATH="./AIRFLOW/dags/sample-dag.py"

echo "Converting docker-compose to Kubernetes manifests..."
kompose convert -f $DOCKER_COMPOSE_PATH -o ./AIRFLOW/k8s/ --volumes hostPath

echo "Deploying Airflow components..."
microk8s kubectl apply -f ./AIRFLOW/k8s/ -n mlops

echo "Waiting for pods to be ready..."
microk8s kubectl wait --for=condition=ready pod -l service=airflow-webserver -n mlops --timeout=300s || true

echo "Copying DAG to the Airflow volume..."
AIRFLOW_POD=$(microk8s kubectl get pods -n mlops -l service=airflow-webserver -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ ! -z "$AIRFLOW_POD" ]; then
  microk8s kubectl cp $DAG_PATH mlops/$AIRFLOW_POD:/opt/airflow/dags/
else
  echo "Warning: Airflow webserver pod not found. DAG not copied."
fi

echo "Services:"
microk8s kubectl get services -n mlops

echo "Run this command in a separate terminal:"
echo "microk8s kubectl port-forward -n mlops service/airflow-webserver 8080:8080 --address 0.0.0.0"

echo "Deployment complete!"