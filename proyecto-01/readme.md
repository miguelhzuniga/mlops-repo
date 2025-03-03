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

## 🚀 Instalación y Configuración  

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
