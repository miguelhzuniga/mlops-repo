# Proyecto numero 3 de MLOps



# Parte 1: Procesamiento de Datos con Airflow

Este repositorio contiene el código necesario para implementar el procesamiento de datos de un conjunto de datos de diabetes utilizando Apache Airflow como herramienta de orquestación.

## Descripción del DAG de Procesamiento

El DAG `diabetes_data_processing` realiza las siguientes tareas:

1. **Preparación**: Crea un directorio temporal para los archivos intermedios.
2. **Descarga de datos**: Obtiene el conjunto de datos de diabetes desde la fuente.
3. **Procesamiento**: Limpia y transforma los datos realizando:
  - Manejo de valores faltantes
  - Codificación de variables categóricas
  - Tratamiento de valores atípicos
  - Ingeniería de características
4. **Almacenamiento de datos crudos**: Guarda los datos originales en PostgreSQL.
5. **División y carga de datos**: Separa los datos en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%), y los almacena en PostgreSQL. El conjunto de entrenamiento se divide en lotes de 15,000 registros.
6. **Limpieza**: Elimina los archivos temporales utilizados durante el proceso.

## Estructura de la Base de Datos

Los datos se almacenan en PostgreSQL con la siguiente estructura:

- **Datos crudos**: `raw_data.diabetes`
- **Datos procesados**:
 - `clean_data.diabetes_train` (dividido en lotes)
 - `clean_data.diabetes_validation`
 - `clean_data.diabetes_test`
 - `clean_data.batch_info` (información sobre los lotes)

## Ejecución

Para ejecutar el DAG, asegúrese de tener configurado Airflow con la conexión a PostgreSQL usando el ID `postgres_default`.

## Próximos Pasos

Este componente es la primera parte de un sistema de MLOps más amplio que incluirá:

1. **Experimentación con MLflow**: Desarrollo y registro de modelos con seguimiento de métricas.
2. **Despliegue de modelos**: Servicio de inferencia a través de FastAPI.
3. **Interfaz gráfica**: Visualización y uso de los modelos mediante Streamlit.
4. **Observabilidad**: Monitoreo con Prometheus y Grafana.
5. **Pruebas de carga**: Evaluación de rendimiento con Locust.

Próximamente se añadirán los DAGs y componentes para estas funcionalidades.

## Requisitos

- Apache Airflow 2.5+
- PostgreSQL 13+
- Python 3.8+
- Bibliotecas: pandas, numpy, scikit-learn, requests

# Parte 2: Plataforma de Experimentación con MLflow

Esta sección permite desplegar MLflow en Kubernetes (MicroK8s) para el seguimiento de experimentos, registro y gestión de modelos.

### Estructura de Archivos

```
├── docker
│   └── Dockerfile.mlflow     # Dockerfile personalizado para MLflow
├── manifests                  # Archivos de configuración Kubernetes
│   ├── ingress.yaml          # Configuración de ingress para acceso web
│   ├── init-job.yaml         # Job para inicializar el bucket de MinIO
│   ├── mlflow.yaml           # Despliegue de MLflow
│   ├── namespace.yaml        # Namespace para el proyecto
│   └── storage.yaml          # Configuración de PostgreSQL y MinIO
└── scripts
    ├── cleanup.sh            # Script para eliminar el despliegue
    └── deploy.sh             # Script para desplegar la infraestructura
```

### Componentes Principales

1. **MLflow Server**: Plataforma para el seguimiento de experimentos, registro de modelos y gestión del ciclo de vida.
2. **PostgreSQL**: Base de datos para almacenar los metadatos de MLflow.
3. **MinIO**: Almacenamiento compatible con S3 para los artefactos de modelos y resultados de experimentos.

### Instrucciones de Despliegue

1. Asegúrese de tener MicroK8s instalado en su sistema:
   ```bash
   sudo snap install microk8s --classic
   sudo usermod -a -G microk8s $USER
   ```

2. Ejecute el script de despliegue:
   ```bash
   chmod +x scripts/deploy.sh
   ./scripts/deploy.sh
   ```

3. El script realizará las siguientes acciones:
   - Habilitar los addons necesarios en MicroK8s
   - Crear el namespace para el proyecto
   - Desplegar PostgreSQL y MinIO
   - Construir y publicar la imagen personalizada de MLflow
   - Desplegar el servidor MLflow
   - Configurar el acceso mediante Ingress

4. Tras la ejecución, podrá acceder a:
   - MLflow UI: http://<NODE_IP>:30500
   - Consola de MinIO: http://<NODE_IP>:30901 (user: adminuser, password: securepassword123)

5. Para eliminar el despliegue cuando ya no sea necesario:
   ```bash
   chmod +x scripts/cleanup.sh
   ./scripts/cleanup.sh
   ```

## Flujo de Trabajo Completo

1. Los datos son procesados mediante el DAG de Airflow
2. Los científicos de datos utilizan MLflow para experimentar con diferentes modelos
3. Los modelos entrenados se registran en MLflow para su seguimiento
4. Los mejores modelos se seleccionan para su despliegue
