# Proyecto numero 3 de MLOps
Este repositorio implementa una plataforma MLOps completa para el análisis, procesamiento, entrenamiento y despliegue de modelos predictivos enfocados en determinar la probabilidad de readmisión hospitalaria de pacientes diabéticos.
La arquitectura del sistema está distribuida en múltiples máquinas virtuales que albergan componentes especializados:

Maquina 01: Orquestación de datos (Airflow) y experimentación (MLflow + JupyterLab)

Maquina 02: Servicios de inferencia (FastAPI + Gradio) accesibles en los puertos 30601 y 30602

Maquina 03: Monitorización y observabilidad (Prometheus + Grafana)

El sistema implementa un flujo de trabajo automatizado desde el procesamiento de datos hasta su división en lotes (batching), experimentación con múltiples modelos, registro y despliegue del mejor modelo en producción, todo sin necesidad de intervención manual durante las transiciones.
La plataforma aprovecha tecnologías de contenedores y Kubernetes para garantizar escalabilidad, portabilidad y despliegue consistente en diferentes entornos.

### Estructura de proyecto
```
├── 01_Primera_maquina
│   ├── airflow
│   │   ├── Dockerfile
│   │   ├── dags
│   │   │   ├── __pycache__
│   │   │   │   ├── proceso_de_datos-dag.cpython-38.pyc
│   │   │   │   └── proceso_mlflow-dag.cpython-38.pyc
│   │   │   ├── proceso_de_datos-dag.py
│   │   │   └── proceso_mlflow-dag.py
│   │   ├── docker-compose.yaml
│   │   ├── logs
│   │   ├── plugins
│   │   └── requirements.txt
│   ├── jupyterlab
│   │   ├── Dockerfile
│   │   ├── experimentos.ipynb
│   │   └── requirements.txt
│   └── mlflow
│       ├── docker
│       │   ├── Dockerfile
│       │   └── Dockerfile.mlflow
│       ├── manifests
│       │   ├── ingress.yaml
│       │   ├── init-job.yaml
│       │   ├── mlflow.yaml
│       │   ├── namespace.yaml
│       │   └── storage.yaml
│       └── scripts
│           ├── cleanup.sh
│           ├── deploy.sh
│           └── docker
│               └── Dockerfile.mlflow
├── 02_Segunda_maquina
│   └── api
│       ├── clean_api.sh
│       ├── deploy.sh
│       ├── fastapi
│       │   ├── Dockerfile
│       │   ├── fastapi-deployment.yaml
│       │   ├── fastapi-service.yaml
│       │   ├── main_server.py
│       │   └── requirements.txt
│       └── gradio
│           ├── Dockerfile
│           ├── gradio-deployment.yaml
│           ├── gradio-service.yaml
│           ├── gradio_app.py
│           └── requirements.txt
├── 03_Tercera_maquina
│   ├── Dockerfile
│   ├── cleanall.sh
│   ├── deploy-monitoring.sh
│   ├── locust
│   │   └── locust.yaml
│   ├── locustfile.py
│   ├── manifests-local
│   │   └── node-specific-deployment.yaml
│   ├── monitoring
│   │   ├── grafana.yaml
│   │   └── prometheus.yaml
│   ├── payload.json
│   └── requirements.txt
├── Imagenes_servicios
│   ├── experimento_locust1.png
│   ├── experimento_locust2.png
│   ├── experimento_locust3.png
│   ├── image-1.png
│   ├── image-1_tercera_maquina.png
│   ├── image-2.png
│   ├── image-2_tercera_maquina.png
│   ├── image-3.png
│   ├── image-3_tercera_maquina.png
│   ├── image-4.png
│   ├── image-5.png
│   ├── image-6.png
│   ├── image-7.png
│   ├── image.png
│   └── image_tercera_maquina.png
└── README.md

```
# **01_Primera_maquina**

# Parte 1: Procesamiento de Datos con Airflow

Esta máquina implementa la capa de procesamiento de datos, experimentación y registro de modelos para el sistema de predicción de readmisiones hospitalarias en pacientes diabéticos.
Arquitectura y Componentes
La máquina alberga tres componentes principales que operan de forma integrada:

Apache Airflow: Orquesta el procesamiento de datos y entrenamiento automatizado

JupyterLab: Proporciona un entorno interactivo para exploración de datos y experimentación

MLflow: Gestiona el registro, versionado y transición a producción de modelos

### Estructura de Directorios
```
─ 01_Primera_maquina
├── airflow
│   ├── Dockerfile                               # Runtime: Python 3.8
│   ├── dags
│   │   ├── pycache
│   │   │   ├── proceso_de_datos-dag.cpython-38.pyc
│   │   │   └── proceso_mlflow-dag.cpython-38.pyc
│   │   ├── proceso_de_datos-dag.py              # DAG de procesamiento ETL
│   │   └── proceso_mlflow-dag.py                # DAG de entrenamiento ML
│   ├── docker-compose.yaml                      # Configuración de servicios
│   ├── logs                                     # Directorio de logs persistentes
│   ├── plugins                                  # Plugins personalizados
│   └── requirements.txt                         # Dependencias Python
├── jupyterlab
│   ├── Dockerfile                               # Runtime: Python 3.10
│   ├── experimentos.ipynb                       # Notebook de experimentación
│   └── requirements.txt                         # Dependencias Python
└── mlflow
├── docker
│   ├── Dockerfile                           # Imagen base
│   └── Dockerfile.mlflow                    # Configuración MLflow
├── manifests
│   ├── ingress.yaml                         # Configuración de acceso
│   ├── init-job.yaml                        # Job de inicialización
│   ├── mlflow.yaml                          # Despliegue del servidor
│   ├── namespace.yaml                       # Namespace Kubernetes
│   └── storage.yaml                         # Configuración de almacenamiento
└── scripts
├── cleanup.sh                           # Script de eliminación
├── deploy.sh                            # Script de despliegue
└── docker
└── Dockerfile.mlflow                # Imagen MLflow

```

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

![alt text](./Imagenes_servicios/image.png)

## Estructura de la Base de Datos

Los datos se almacenan en PostgreSQL con la siguiente estructura:

- **Datos crudos**: `raw_data.diabetes`
- **Datos procesados**:
 - `clean_data.diabetes_train` (dividido en lotes)
 - `clean_data.diabetes_validation`
 - `clean_data.diabetes_test`
 - `clean_data.batch_info` (información sobre los lotes)

**Nota:** Para realizar monitoreo o consulta a la base sin necesidad de realizar un script, se ha disponibilizado el servicio PGadmin para poder consultar cada las bases de raw.data y clean.data

![alt text](./Imagenes_servicios/image-1.png)


## Experimentos con servicio Jupyterlab

Este componente es la sección de prueba por la exploración de datos y testeo de nuevos modelos.

Experimentos con servicio JupyterLab
El servicio JupyterLab opera como laboratorio de experimentación y análisis de datos, proporcionando un entorno interactivo para científicos de datos e investigadores.
Características técnicas

Imagen base customizada: Construida sobre jupyter/base-notebook:python-3.10 con dependencias preinstaladas
Endpoint accesible: Expuesto en puerto 8888 sin autenticación para facilitar integración
Configuración optimizada: Memoria compartida con kernels para procesamiento eficiente
Dependencias precargadas: MLflow, scikit-learn, LightGBM, pandas y librerías de visualización

![alt text](./Imagenes_servicios/image-2.png)
### **Flujo de experimentos implementados**

Exploración de datos: Análisis estadístico descriptivo de variables clínicas y demográficas

Preprocesamiento: Pipeline personalizado con StandardScaler para variables numéricas y OneHotEncoder para categóricas

Evaluación comparativa: Benchmark de modelos (LogisticRegression, DecisionTree) con tracking automatizado

Optimización de hiperparámetros: Grid search manual con registro MLflow de cada configuración

Interfaz de predicción: Función predict_readmission() para validar modelos con casos de prueba

### **Integración con componentes MLOps**

Conexión PostgreSQL: 
postgresql://airflow:airflow@localhost:5432/airflow para acceso a datos procesados

Tracking MLflow: Configurado en http://10.43.101.175:30500 para registro de experimentos y modelos
Almacenamiento S3: MinIO (http://10.43.101.175:30382) para persistencia de artefactos y preprocesadores

Visualización integrada: Gráficos, matrices de confusión y métricas exportadas como artefactos MLflow

El notebook experimentos.ipynb funciona como plantilla reproducible para futuras iteraciones y pruebas de concepto, manteniendo consistencia con los pipelines productivos.

## **Registros de modelos y experimentos por MLFLOW**

## Estructura de archivos

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


Registros de modelos y experimentos por MLflow
MLflow opera como componente central de registro y gobernanza de modelos, desplegado en Kubernetes con endpoints específicos:

- Tracking Server: Accesible en puerto 30500, proporciona interfaz web para visualización y comparación de experimentos
- Backend PostgreSQL: Almacena metadatos, parámetros, métricas y linaje de modelos
- Almacenamiento S3 (MinIO): Gestiona artefactos (modelos, preprocesadores, visualizaciones) en el bucket mlflow-artifacts

Estructura de experimentos implementada

Experimento principal: diabetes_readmission_experiment contiene todas las ejecuciones de modelos
Jerarquía de ejecuciones: Las ejecuciones anidadas permiten agrupar modelos y optimizaciones relacionadas
Etiquetas de transición: (Production, Staging, Archived) facilitan el despliegue automático

![alt text](./Imagenes_servicios/image-3.png)
![alt text](./Imagenes_servicios/image-4.png)

## REQUISITOS PARA LA IMPLEMENTACIÓN EN MAQUINA 1

# > Librerias

- Apache Airflow 2.5+
- PostgreSQL 13+
- Python 3.8+
- Bibliotecas: pandas, numpy, scikit-learn, requests

## > Especificaciones Técnicas

### Apache Airflow (Puerto 8080)

- **Contenedores**:
  - Webserver (8080)
  - Scheduler
  - Worker
  - Triggerer
  - PostgreSQL (5432)
  - Redis (6379)
  - Inicializador
  - CLI (debug)
  - Flower (5555, opcional)
  - PgAdmin (5050)
  - Network-tools

- **DAGs implementados**:
  - `diabetes_data_processing`: Procesa datos de readmisión hospitalaria (ejecución diaria + 1h)
  - `diabetes_ml_pipeline_optimized`: Entrena y registra modelos de ML (ejecución semanal)

- **Optimizaciones de recursos disponibles**:
  - Control de hilos para LightGBM (n_jobs=1)
  - Muestreo de datos (10% para entrenamiento)
  - Timeout de tareas configurable (3600s)

- **Conexiones externas**:
  - MLflow: `http://10.43.101.175:30500`
  - MinIO: `http://10.43.101.175:30382`
  - Credenciales S3: adminuser/securepassword123

### JupyterLab (Puerto 8888)

- **Imagen base**: `jupyter/base-notebook:python-3.10`
- **Notebooks**:
  - `experimentos.ipynb`: Implementa exploración, preprocesamiento, modelado y evaluación
- **Acceso**: Sin token/contraseña para facilitar integración
- **Características**:
  - Visualización con matplotlib/seaborn
  - Integración con MLflow para registro de experimentos
  - Conexión a PostgreSQL (`postgresql://airflow:airflow@localhost:5432/airflow`)

### MLflow (Puerto 30500)

- **Despliegue**: Kubernetes (MicroK8s)
- **Componentes**:
  - Tracking Server (30500)
  - PostgreSQL para metadatos
  - MinIO para artefactos (30382, 30901)
- **Almacenamiento**:
  - Bucket: `mlflow-artifacts`
  - Ruta de preprocesadores: `preprocessors/preprocessor.joblib`
  - Modelos registrados: `diabetes_readmission_{model_name}`

## **Flujo de Trabajo**

1. El DAG `diabetes_data_processing` ejecuta ETL del conjunto de datos:
   - Descarga datos de pacientes diabéticos
   - Procesa y limpia los datos
   - Divide en conjuntos de entrenamiento, validación y test
   - Particiona los datos de entrenamiento en lotes (batch_size=15000)

2. JupyterLab permite a científicos de datos experimentar con:
   - Exploración de distribuciones y características
   - Preprocesamiento avanzado (normalización, codificación)
   - Entrenamiento de modelos (LogisticRegression, DecisionTree)
   - Optimización de hiperparámetros
   - Análisis de métricas (precisión, recall, F1)

3. El DAG `diabetes_ml_pipeline_optimized` ejecuta automáticamente:
   - Carga incremental de lotes de datos
   - Preprocesamiento consistente
   - Entrenamiento de múltiples modelos (LightGBM, DecisionTree, LogisticRegression)
   - Evaluación comparativa
   - Registro del mejor modelo en MLflow
   - Transición a etapa "Production" del modelo ganador

## > Requisitos del Sistema

- Docker y Docker Compose
- Kubernetes (MicroK8s)
- 4GB RAM mínimo recomendado
- 2 CPUs mínimo recomendados
- 10GB espacio en disco

## > Configuración de Red

| Servicio | Puerto | Protocolo | Descripción |
|----------|--------|-----------|-------------|
| Airflow Webserver | 8080 | HTTP | Interfaz de administración de DAGs |
| PostgreSQL | 5432 | TCP | Base de datos para Airflow y datos procesados |
| PgAdmin | 5050 | HTTP | Interfaz de administración de PostgreSQL |
| Redis | 6379 | TCP | Backend para ejecutor Celery |
| Flower | 5555 | HTTP | Monitoreo de tareas Celery (opcional) |
| JupyterLab | 8888 | HTTP | Entorno de experimentación |
| MLflow | 30500 | HTTP | Tracking server y registro de modelos |
| MinIO API | 30382 | HTTP | API S3 para artefactos |
| MinIO Console | 30901 | HTTP | Interfaz web de MinIO |

## Instrucciones de Despliegue

1. **Airflow y JupyterLab**:
   ```bash
   cd 01_Primera_maquina/airflow
   docker-compose up -d
2. **Mlflow** 
```bash
cd 01_Primera_maquina/mlflow/scripts
./deploy.sh
```


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

# **02_Segunda_maquina**

La carpeta `02_Segunda_maquina` contiene la implementación de la interfaz de inferencia y la API de predicción para el modelo de diabetes. Esta sección del proyecto implementa el servicio de inferencia que consume los modelos entrenados en MLflow.

### Estructura de archivos
```
└── api/
├── clean_api.sh              # Script de desmontaje de recursos K8s
├── deploy.sh                 # Automatización CI/CD para K8s
├── fastapi/                  # Backend API REST
│   ├── Dockerfile            # Runtime Python 3.9-slim
│   ├── fastapi-deployment.yaml
│   ├── fastapi-service.yaml  # NodePort: 30601
│   ├── main_server.py        # Implementación del servicio
│   └── requirements.txt      # Dependencias
└── gradio/                   # Frontend interactivo
├── Dockerfile            # Runtime Python 3.8.20
├── gradio-deployment.yaml
├── gradio-service.yaml   # NodePort: 30602
├── gradio_app.py         # Implementación de UI
└── requirements.txt      # Dependencias

```

### Características principales

- **Conexión automática con MLflow**: Ambos servicios obtienen el modelo de producción de forma dinámica desde MLflow, sin necesidad de recodificar o recompilar al cambiar de modelo.

- **Robustez en el procesamiento de datos**: Incluye manejo especial para valores atípicos, campos con comparadores (">200", "<30"), y normalización de nombres de campos para asegurar compatibilidad con el preprocesador original 

 **Nota:** Este proceso se realiza por la capacidad limita de las maquinas virtuales, en un entorno de producción con mayores recursos, se puede disponer de un entrenamiento mas profundo.

- **Exposición de métricas para Prometheus**: Ambos servicios exponen endpoints `/metrics` para recolección de métricas como número de solicitudes, errores, tiempos de respuesta, etc.

- **Contenedorización completa**: Todos los componentes están containerizados y configurados para despliegue en Kubernetes, facilitando la escalabilidad y gestión.

### Flujo de trabajo

1. El script `deploy.sh` construye las imágenes Docker para FastAPI y Gradio
2. Las imágenes se etiquetan y suben a Docker Hub
3. Los manifiestos de Kubernetes se aplican para crear los despliegues y servicios
4. El servicio FastAPI (backend) proporciona una API REST para realizar predicciones
5. La interfaz Gradio (frontend) ofrece una experiencia amigable para usuarios finales
6. Ambos servicios consultan MLflow para obtener siempre el modelo marcado como "Production"

### Endpoints y conectividad

- **FastAPI (Backend)**: `http://10.43.101.206:30601`
  - `/predict`: POST - Endpoint principal para inferencia
  - `/health`: GET - Verificación de salud del servicio
  - `/metrics`: GET - Exposición de métricas para 

![alt text](./Imagenes_servicios/image-5.png)
![alt text](./Imagenes_servicios/image-6.png)
![alt text](./Imagenes_servicios/image-7.png)

- **Dependencias externas**:
  - MLflow Server: `http://10.43.101.206:30500`
  - MinIO S3: `http://10.43.101.206:30382`
  - Credenciales S3: adminuser/securepassword123

### Características técnicas

- **Orquestación K8s**: Ambos servicios utilizan Kubernetes para gestión de contenedores, definiendo límites de recursos (CPU: 500m, Memoria: 1Gi) y solicitudes (CPU: 200m, Memoria: 512Mi).

- **Gestión dinámica de modelos**: Implementación de modelo Registry Pattern conectado a MLflow, permitiendo promoción/degradación de modelos sin redeployment.

- **Instrumentación**: Integración con Prometheus mediante cliente nativo exponiendo métricas REQUESTS, PREDICTIONS, PREDICTION_TIME, MODEL_ERRORS y PREPROCESSOR_ERRORS.

- **Procesamiento de datos**: Pipeline de transformación con soporte para manejo de valores atípicos, normalización de nombres de campos y conversión automática de tipos.

- **Almacenamiento en caché**: Implementación de patrón Singleton para modelos y preprocesadores, minimizando latencia y consumo de recursos.

**Nota:**  La funcionalidad de recolección de datos de inferencia se encuentra temporalmente deshabilitada en el código (comentada). Esto se debe a que la arquitectura actual, con Airflow operando como servicio en Docker Compose, introduce una latencia significativa en las conexiones a la base de datos que compromete la experiencia del usuario final. Dado que esta recolección no es crítica en la fase actual del proyecto, se ha preservado la implementación del flujo de trabajo pero desactivado su ejecución. Para resolver esta limitación técnica, se recomienda migrar el despliegue de Airflow a Kubernetes, lo que permitiría una comunicación más eficiente entre la API de inferencia y el esquema raw.data en PostgreSQL.

# **03_Tercera_maquina**

Esta máquina implementa la capa de monitorización y observabilidad para la plataforma MLOps, proporcionando capacidades de pruebas de carga, recolección de métricas en tiempo real y visualización de telemetría operacional.

## Estructura de directorios
```
─ 03_Tercera_maquina
├── cleanall.sh                # Script de terminación de recursos K8s (namespace mlops-project)
├── deploy-monitoring.sh       # Script CI/CD para K8s con detección de nodo
├── locust/                    # Framework de pruebas de carga distribuidas
│   ├── Dockerfile             # Extensión de imagen base locustio/locust
│   ├── locustfile.py          # Implementación de casos de prueba sintéticos
│   ├── locust.yaml            # Manifiesto K8s (NodePort: 31000, API Target: 10.43.101.202:30601)
│   └── payload.json           # Payload de referencia para simulación de inferencia
└── monitoring/
├── prometheus.yaml        # Configuración scraper con job_name: 'fastapi' (NodePort: 31090)
└── grafana.yaml           # Panel de visualización con datasource Prometheus (NodePort: 31300)

```
## Componentes de observabilidad

- **Locust**: Implementa testing de carga distribuido con arquitectura master-worker, configurado para generar tráfico sintético hacia el endpoint de inferencia REST (`10.43.101.202:30601/predict`). La simulación replica patrones de tráfico de usuarios reales con distribuciones de espera entre 1-5 segundos y payload completo para evaluar la serialización/deserialización bajo carga.

- **Prometheus**: Servicio de monitorización basado en time-series database que implementa pull-based metrics collection con intervalos de scraping de 15s. Configurado para recolectar métricas del endpoint `/metrics` expuesto por FastAPI (PORT 30601) con instrumentación personalizada para tracking de:
  - `prediction_requests_total`: Contador acumulativo de solicitudes
  - `prediction_latency_seconds`: Histograma con bucketing automático para análisis de percentiles
  - `prediction_errors_total`: Contador segregado por código de error/excepción

- **Grafana**: Plataforma de visualización implementada en modo stateless, preconfigurable mediante provisioning. Expone el puerto 31300 mediante NodePort y utiliza Prometheus como datasource principal. Soporta configuración declarativa de dashboards mediante ConfigMaps para facilitar GitOps.

## Metricas operacionales claves

El sistema de monitorización está instrumentado para capturar las siguientes métricas críticas:

| Métrica | Tipo | Descripción | Rango esperado |
|---------|------|-------------|----------------|
| Latencia de predicción P95 | Histograma | Tiempo de respuesta del endpoint `/predict` (percentil 95) | <500ms |
| Throughput | Contador | Solicitudes por segundo procesadas | 1-100 RPS |
| Tasa de errores | Ratio | Porcentaje de solicitudes con código 4xx/5xx | <0.1% |
| Utilización CPU | Gauge | Porcentaje de CPU utilizado por pod de inferencia | <80% |
| Memoria utilizada | Gauge | Consumo de memoria del pod de inferencia | <1.5GiB |
| Tiempo de carga de modelo | Histograma | Duración de carga inicial de modelos | <2s |

## Despliegue y conexión

El stack completo se despliega en el namespace `mlops-project` dentro del clúster MicroK8s y es accesible mediante NodePorts:

- **Locust UI**: `http://10.43.101.202:31000` - Interfaz para pruebas de carga
- **Prometheus**: `http://10.43.101.202:31090` - Plataforma de métricas con PromQL
- **Grafana**: `http://10.43.101.202:31300` - Panel de visualización (admin/admin)

Para iniciar el despliegue completo:

```bash
chmod +x deploy-monitoring.sh
sudo ./deploy-monitoring.sh
```

## Para desmontar todos los recursos:


```bash
chmod +x cleanall.sh
sudo ./cleanall.sh
```
## **Integración end-to-end en el ecosistema MLOps**

Este componente de monitorización completa el ciclo MLOps proporcionando observabilidad integral:

- Validación de modelos bajo carga: Cuantificación del degradado de rendimiento del modelo en función del throughput

- Detección de data drift: Instrumentación para capturar anomalías estadísticas en distribuciones de entrada

- Dimensionamiento de recursos: Métricas para optimización de límites/solicitudes de CPU/memoria en pods de FastAPI

- Circuit-breaking proactivo: Capacidad de establecer umbrales de alerta para degradación de servicio

- Dashboards operacionales: Visualización consolidada de métricas técnicas y de negocio (precision/recall)

- La integración con el sistema de experimentación MLflow (maquina 1) y el servicio de inferencia FastAPI (maquina 2) cierra el ciclo de feedback para mejora continua de los modelos.

# **Visualización del monitoreo**
# Prometheus
![alt text](./Imagenes_servicios/image_tercera_maquina.png)
# Locust UI
![alt text](./Imagenes_servicios/image-1_tercera_maquina.png)
![alt text](./Imagenes_servicios/image-2_tercera_maquina.png)
## Experimentos Locust UI
# Exp1
![alt text](./Imagenes_servicios/experimento_locust1.png)
# Exp2
![alt text](./Imagenes_servicios/experimento_locust2.png)
# Exp3
![alt text](./Imagenes_servicios/experimento_locust3.png)

# Grafana
![alt text](./Imagenes_servicios/image-3_tercera_maquina.png)

# Conclusión del monitoreo
Locust UI: Nos permitió identificar que la arquitectura actual soporta un máximo de 50 usuarios simultáneos, incrementando 5 usuarios por segundo. Este límite se determina porque el tiempo máximo de respuesta para las predicciones alcanza los 4.9 segundos, lo cual consideramos aceptable dentro de un umbral de hasta 5 segundos. Superar este número de usuarios generaría tiempos de respuesta mayores al límite establecido.

Prometheus & Grafana: Estas herramientas nos permitieron recolectar métricas específicas de nuestras APIs, como la cantidad de predicciones realizadas y el tiempo de ejecución de cada una. Esta información resulta muy útil y puede visualizarse de manera clara y atractiva a través de los dashboards de Grafana.

## Conclusión General del Proyecto MLOps en Kubernetes
Este proyecto implementa una plataforma MLOps completa y robusta utilizando Kubernetes como base de orquestación para gestionar el ciclo de vida de modelos predictivos enfocados en la readmisión hospitalaria de pacientes diabéticos.

Arquitectura Distribuida y Componentes Clave
La arquitectura está estratégicamente distribuida en tres máquinas virtuales, cada una con responsabilidades específicas y claramente definidas:

- Máquina 1 (Procesamiento y Experimentación): Implementa Airflow para orquestación de datos, JupyterLab para experimentación y MLflow para registro de modelos, creando una base sólida para el desarrollo y preparación de modelos.

- Máquina 2 (Servicios de Inferencia): Proporciona una API REST (FastAPI) y una interfaz de usuario (Gradio) para realizar predicciones en tiempo real, con capacidad de cargar dinámicamente el mejor modelo desde MLflow.

- Máquina 3 (Monitorización): Despliega Prometheus, Grafana y Locust para recolectar métricas, visualizar rendimiento y realizar pruebas de carga, cerrando el ciclo de retroalimentación para la mejora continua.

## Fortalezas Técnicas
El proyecto demuestra varias fortalezas técnicas significativas:

- Automatización completa: Desde el procesamiento de datos hasta el despliegue en producción, minimizando la intervención manual y reduciendo errores.

- Containerización y orquestación: El uso de Docker y Kubernetes garantiza despliegues consistentes, portabilidad y escalabilidad.

- Registro y versión de modelos: MLflow proporciona trazabilidad completa de experimentos, parámetros y métricas, facilitando la reproducibilidad y comparación.

- Despliegue dinámico: La implementación del patrón Registry permite promocionar o degradar modelos sin necesidad de redespliegue.

- Observabilidad integral: La instrumentación con Prometheus permite monitorizar tanto aspectos técnicos (latencia, throughput) como de negocio (precisión del modelo).

# Capacidades Operacionales
Las pruebas de carga con Locust revelaron que la arquitectura actual soporta eficientemente hasta 50 usuarios concurrentes con tiempos de respuesta menores a 5 segundos, estableciendo un límite claro para el dimensionamiento de recursos.
La integración entre las tres máquinas crea un ciclo completo de MLOps que abarca desde el procesamiento de datos hasta la monitorización en producción, permitiendo:

# Detección de data drift
Optimización de recursos computacionales
Visualización consolidada de métricas operativas
Evaluación de modelos bajo condiciones de carga real

# Consideraciones Futuras
Para entornos con mayores exigencias, sería recomendable:

Migrar Airflow de Docker Compose a Kubernetes para mejorar la comunicación con la API de inferencia
Implementar autoescalado basado en métricas para adaptarse a picos de demanda
Habilitar la recolección de datos de inferencia para monitorizar el desempeño del modelo a largo plazo

Esta implementación demuestra efectivamente cómo Kubernetes puede proporcionar la infraestructura necesaria para desplegar flujos de trabajo de machine learning completos, desde el desarrollo hasta la producción, con observabilidad y escalabilidad incorporadas.

# Autores:
- **Luis Frontuso**
- **Miguel Zuñiga**
- **Camilo Serrano**
-----