# Poryecto-01: Entorno de Desarrollo para Machine Learning  

Este proyecto tiene como objetivo construir un ambiente de desarrollo para machine learning, que permita la ingesta, validación y transformación de datos. Para esto se creo un ambiente de desarrollo de jupyterlab en un contenedor docker basado en la imagen python:3.9-slim-bookworm, este ambiente se despliega en el escritorio de la máquina virtual **10.43.101.206**.

## 📁 Estructura del Repositorio

```plaintext
proyecto-01/
├── train/                      # 📁 Carpeta principal del entrenamiento
│   ├── Dockerfile              # 🐳 Configuración del contenedor de entrenamiento
│   ├── __pycache__/            # 🗄️ Caché de archivos compilados
│   ├── data/                   # 📂 Conjunto de datos
│   │   ├── covertype/          
│   │   │   ├── covertype_train.csv   # 📄 Datos de entrenamiento
│   │   │   ├── serving.csv           # 📄 Datos de prueba
│   │   ├── data_new/         
│   │   │   ├── cover_new.csv         # 📄 Datos nuevos
│   │   ├── data_service/     
│   │   │   ├── service_data.csv      # 📄 Datos del servicio
│   ├── metadata/               # 🏛️ Almacenamiento de metadatos
│   │   ├── metadata.sqlite
│   │   ├── metadata.db
│   ├── pipeline_outputs/        # 🔄 Salida del pipeline de datos
│   │   ├── CsvExampleGen/
│   │   ├── _wheels/
│   ├── pipeline_root/           # 🏗️ Raíz del pipeline
│   ├── schema/                  # 📑 Esquemas del dataset
│   │   ├── schema.pb
│   │   ├── schema.pbtxt
│   │   ├── schema_with_envs.pbtxt
│   ├── data_preparation.py       # 🔢 Preparación de datos
│   ├── model_creation.py         # 🔍 Creación del modelo
│   ├── preprocessing.py          # ⚙️ Preprocesamiento de datos
│   ├── proyecto-01.ipynb         # 📊 Notebook con el desarrollo del proyecto
│   ├── requirements.txt          # 📦 Dependencias del proyecto
│
├── docker-compose.yml            # 🐳 Configuración de Docker
├── jupyterlab.service            # 🔧 Configuración del servicio JupyterLab
├── jupyterlab.sh                 # 🖥️ Script para iniciar JupyterLab
├── readme.md                     # 📜 Documentación del proyecto
```
## 🔹 Características del Proyecto  

El entorno de desarrollo construido permitirá ejecutar un flujo de datos con las siguientes etapas:  
- **Selección de características**  
- **Ingesta del conjunto de datos**  
- **Generación de estadísticas del dataset**  
- **Creación de un esquema basado en el conocimiento del dominio**  
- **Definición y creación de entornos de esquema**  
- **Visualización de anomalías en los datos**  
- **Preprocesamiento, transformación e ingeniería de características**  
- **Seguimiento de la procedencia del flujo de datos mediante metadatos de ML**  

## 📌 Tecnologías y Herramientas  

- **Docker** y **Docker Compose** para la configuración y aislamiento del entorno.  
- **Jupyter Notebook** como interfaz interactiva para la ejecución del código.  
- **Git y GitHub** para el control de versiones.  
- **Bash script y systemd** para la disponibilización del ambiente cada vez que se inicie la VM.
- **Dataset**: *Tipo de Cubierta Forestal*  

| Column Name                                    | Variable Type  | Units / Range                        | Description                                      |
|------------------------------------------------|---------------|--------------------------------------|--------------------------------------------------|
| Elevation                                      | quantitative  | meters                               | Elevation in meters                             |
| Aspect                                         | quantitative  | azimuth                              | Aspect in degrees azimuth                       |
| Slope                                          | quantitative  | degrees                              | Slope in degrees                                |
| Horizontal_Distance_To_Hydrology               | quantitative  | meters                               | Horz Dist to nearest surface water features     |
| Vertical_Distance_To_Hydrology                 | quantitative  | meters                               | Vert Dist to nearest surface water features     |
| Horizontal_Distance_To_Roadways                | quantitative  | meters                               | Horz Dist to nearest roadway                   |
| Hillshade_9am                                  | quantitative  | 0 to 255 index                       | Hillshade index at 9am, summer solstice         |
| Hillshade_Noon                                 | quantitative  | 0 to 255 index                       | Hillshade index at noon, summer solstice        |
| Hillshade_3pm                                  | quantitative  | 0 to 255 index                       | Hillshade index at 3pm, summer solstice         |
| Horizontal_Distance_To_Fire_Points             | quantitative  | meters                               | Horz Dist to nearest wildfire ignition points   |
| Wilderness_Area (4 binary columns)            | qualitative   | 0 (absence) or 1 (presence)          | Wilderness area designation                     |
| Soil_Type (40 binary columns)                 | qualitative   | 0 (absence) or 1 (presence)          | Soil Type designation                          |
| Cover-Type (7 types)                           | integer       | 1 to 7                               | Forest Cover Type designation                   |

# Pipeline TFX para Tipo de Cobertura Forestal ![JupyterLab](https://jupyter.org/assets/homepage/main-logo.svg)

## Descripción General
Este proyecto implementa un pipeline de TensorFlow Extended (TFX) para predecir tipos de cobertura forestal utilizando el conjunto de datos UCI Covertype. El pipeline abarca procesos de ingesta de datos, validación, transformación, ingeniería de características y entrenamiento de modelos siguiendo las mejores prácticas de MLOps.

## Requisitos Previos
- Python 3.7+
- TensorFlow 2.x
- TensorFlow Extended (TFX)
- TensorFlow Data Validation (TFDV)
- Pandas
- Scikit-learn
- ML Metadata

## Estructura del Proyecto
```
├── data/
│   ├── covertype/         # Almacenamiento de datos sin procesar
│   ├── data_new/          # Datos con selección de características
│   └── data_service/      # Datos de prueba del entorno de servicio
├── schema/                # Definiciones de esquema
├── preprocessing.py       # Módulo de transformación de características
├── data_preparation.py    # Funciones de carga y preprocesamiento de datos
├── model_creation.py      # Funciones de definición de modelos
├── README.md              # Este archivo
```

## Componentes del Pipeline

1. **Ingesta de Datos**
   - Descarga el conjunto de datos Covertype desde UCI
   - Carga y prepara los datos para el procesamiento

2. **Selección de Características**
   - Utiliza SelectKBest con f_classif para seleccionar las 8 características más relevantes
   - Guarda el conjunto de datos reducido para su uso posterior

3. **Generación de Ejemplos**
   - Utiliza CsvExampleGen para convertir los datos en formato TFRecord

4. **Generación de Estadísticas**
   - Calcula estadísticas descriptivas para todas las características
   - Visualiza las distribuciones de datos

5. **Generación e Inferencia de Esquema**
   - Infiere automáticamente un esquema a partir de los datos
   - Cura el esquema manualmente para definir dominios específicos para características críticas
   - Define entornos de TRAINING y SERVING para gestionar diferencias entre ambos

6. **Validación de Ejemplos**
   - Detecta anomalías en los datos respecto al esquema definido

7. **Transformación de Características**
   - Implementa diferentes estrategias de escalado para las características numéricas:
     - Escalado a rango [0, 1]
     - Escalado Min-Max
     - Normalización Z-score

8. **Inspección de Datos Transformados**
   - Obtiene ejemplos transformados y el gráfico de transformación
   - Lee y muestra ejemplos de TFRecord para verificar el formato de los datos transformados

9. **Metadatos de ML**
   - Almacena y rastrea todos los artefactos generados durante el pipeline
   - Permite la visualización del linaje de datos
   - Incluye funciones auxiliares para:
     - Mostrar tipos de artefactos disponibles
     - Mostrar artefactos específicos como esquemas
     - Mostrar propiedades de artefactos
     - Obtener artefactos principales que alimentaron otros artefactos

## Uso

Para ejecutar el pipeline completo:

```python
# Iniciar el contexto interactivo
context = InteractiveContext(pipeline_root=pipeline_root, 
                             metadata_connection_config=metadata_config)

# Ejecutar los componentes secuencialmente
context.run(example_gen)
context.run(statistics_gen)
context.run(schema_gen)
context.run(import_schema)
context.run(example_validator)
context.run(transform)

# Obtener los datos transformados y el gráfico de transformación
transformed_examples_channel = transform.outputs['transformed_examples']
transform_graph_channel = transform.outputs['transform_graph']

# Inspeccionar los datos transformados (ejemplos)
train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'Split-train')
tfrecord_filenames = [os.path.join(train_uri, name) for name in os.listdir(train_uri)]
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

# Verificar metadatos
store = mlmd.MetadataStore(metadata_config)
display_types(store.get_artifact_types())
schema_artifacts = store.get_artifacts_by_type("Schema")
display_artifacts(store, schema_artifacts)
```

## Funcionalidades de Metadatos

El pipeline incluye varias funciones auxiliares para trabajar con metadatos:

- `display_types()`: Muestra todos los tipos de artefactos disponibles en el almacén
- `display_artifacts()`: Muestra información sobre artefactos específicos
- `display_properties()`: Muestra propiedades detalladas de un artefacto
- `get_one_hop_parent_artifacts()`: Obtiene los artefactos padre de un artefacto determinado

Estas funciones facilitan el seguimiento del linaje de datos a lo largo del pipeline.

## Notas Adicionales

- El pipeline está diseñado para funcionar en modo interactivo para desarrollo y experimentación
- Los esquemas definidos consideran diferentes entornos (entrenamiento vs. servicio)
- Se implementan transformaciones de datos reutilizables a través del módulo `preprocessing.py`
- Se utiliza ML Metadata para el seguimiento de artefactos y linaje de datos

## 🚀 Instalación y Configuración  mediante maquina virtual

### Prerrequisitos de la VM
Asegúrate de tener instalado en tu sistema:  
- [Docker](https://docs.docker.com/get-docker/)  
- [Docker Compose](https://docs.docker.com/compose/install/)  
- [Git](https://git-scm.com/)  

### Clonar el repositorio  
```bash
cd /home/estudiante/Desktop
git clone https://github.com/miguelhzuniga/mlops-repo.git
cd mlops-repo/proyecto-01/
```
### Crear y habilitar el servicio en VM

```bash
sudo cp jupyterlab.service /etc/systemd/system/jupyterlab.service 
sudo chmod +x jupyterlab.sh
sudo systemctl daemon-reload
sudo systemctl enable myscript.service
sudo systemctl start myscript.service
```
## 🖥️ Instrucciones de uso
Iniciar la VM y loguearse.
```bash
ssh -L 8888:localhost:8888 estudiante@10.43.101.206
```
Abrir el entorno de desarrollo.
 - [Jupyterlab](http://localhost:8888/lab)  
