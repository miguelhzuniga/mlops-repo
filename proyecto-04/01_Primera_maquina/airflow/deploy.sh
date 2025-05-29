#!/bin/bash
cleanup() {
    echo "Limpiando instalación fallida..."
    sudo microk8s helm3 uninstall airflow -n mlops-project 2>/dev/null || true
}
trap cleanup ERR
echo "Construyendo imagen personalizada de Airflow..."
cd docker
if sudo docker build -t luisfrontuso10/airflow-custom:latest .; then
    echo "Imagen construida exitosamente"
else
    echo "Error construyendo imagen"
    exit 1
fi
cd ..
echo "Subiendo imagen a Docker Hub..."
if sudo docker push luisfrontuso10/airflow-custom:latest; then
    echo "Imagen subida exitosamente a Docker Hub"
else
    echo "Error subiendo imagen. ¿Hiciste 'docker login'?"
    exit 1
fi
echo "Agregando repositorio Helm..."
sudo microk8s helm3 repo add apache-airflow https://airflow.apache.org
sudo microk8s helm3 repo update
echo "Limpiando instalación previa..."
sudo microk8s helm3 uninstall airflow -n mlops-project 2>/dev/null || true
sleep 10
echo "Instalando Airflow..."
if sudo microk8s helm3 install airflow apache-airflow/airflow \
    -n mlops-project \
    -f values.yaml \
    --timeout 15m; then
    echo "Airflow instalado correctamente!"
else
    echo "Error: Instalación de Helm falló"
    cleanup
    exit 1
fi
echo "Ejecutando migraciones de base de datos..."
sleep 30  # Esperar que PostgreSQL esté listo
if sudo microk8s kubectl run airflow-db-upgrade \
    --image=luisfrontuso10/airflow-custom:latest \
    --rm -i --restart=Never \
    -n mlops-project \
    --env="AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow123@airflow-postgresql.mlops-project:5432/airflow" \
    -- airflow db upgrade; then
    echo "Migraciones completadas exitosamente"
else
    echo "Error en migraciones, pero continuando..."
fi
echo "Configurando conexión PostgreSQL..."
sleep 10
sudo microk8s kubectl exec -it airflow-scheduler-$(sudo microk8s kubectl get pods -n mlops-project -l component=scheduler -o jsonpath='{.items[0].metadata.name}') -n mlops-project -- \
  airflow connections add postgres_default \
  --conn-type postgres \
  --conn-host airflow-postgresql.mlops-project \
  --conn-login airflow \
  --conn-password airflow123 \
  --conn-port 5432 \
  --conn-schema airflow 2>/dev/null || echo "Conexión ya existe"
echo "Verificando estado de los pods..."
sudo microk8s kubectl get pods -n mlops-project
echo ""
echo "Instalación completada exitosamente!"
echo "Accede en: http://localhost:30080"
echo "Usuario: airflow"
echo "Contraseña: airflow"
echo ""
echo "Comandos útiles:"
echo "  Ver pods: sudo microk8s kubectl get pods -n mlops-project"
echo "  Ver logs: sudo microk8s kubectl logs -f deployment/airflow-scheduler -n mlops-project"