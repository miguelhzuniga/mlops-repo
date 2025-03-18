# 🚀 MLOps Stack con MinIO, PostgreSQL, FastAPI y pgAdmin

## Acceso a MV

Primero se debe acceder a la máquina virtual NOAM11

•	Usuario escritorio remoto: estudiante
•	Contraseña: 1007710896*/LFg
•	Dirección IP: 10.43.101.175

## Docker compose

En el archivo `docker-compose.yml` se define un entorno completo para experimentación con **MLOps**, que incluye almacenamiento S3, base de datos PostgreSQL, una API en FastAPI y una interfaz de administración de base de datos.

Para empezar se debe ejecutar:

   ```bash
   sudo docker compose up -d
   ```

## 🏗️ Servicios

### **1. MinIO (Almacenamiento tipo S3)**
- Implementa un servidor compatible con **Amazon S3** para almacenar artefactos de modelos.
- Expone:
  - `9000`: API S3 para almacenamiento de archivos.
  - `9001`: Consola web para gestionar los datos.
- **Credenciales predeterminadas**:
  - Usuario: `admin`
  - Contraseña: `supersecret`
- **Datos persistentes en `./minio`**.

### **2. PostgreSQL (Base de datos para experimentos)**
- Almacena metadatos de experimentos de MLFlow.
- Expone el puerto `5432`.
- **Credenciales predeterminadas**:
  - Usuario: `user`
  - Contraseña: `password`
  - Base de datos: `experiments`
- **Datos persistentes en `./postgres_data`**.

### **3. FastAPI (API para servir modelos)**
- API en **FastAPI** para gestionar modelos de MLFlow y hacer inferencias.
- Conectada a PostgreSQL y a un servidor de **MLFlow Tracking** (`MLFLOW_TRACKING_URI`).
- Expone el puerto `8000`.
- Instalación de dependencias en el contenedor:
  - `fastapi`, `uvicorn`, `mlflow`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `boto3`, entre otras.

### **4. pgAdmin (Interfaz web para PostgreSQL)**
- Interfaz gráfica para administrar la base de datos PostgreSQL.
- Expone el puerto `5050`.
- **Credenciales predeterminadas**:
  - Email: `admin@example.com`
  - Contraseña: `admin`.

---

## ⚡ MLflow Tracking Server

El servidor de **MLflow Tracking** se configura como un servicio de `systemd` para ejecutarse en segundo plano y reiniciarse en caso de fallo.

## ⚙️ Configuración de systemd service

Para gestionar el servicio de **MLflow Tracking Server** mediante `systemd`, se debe seguir estos pasos:

1. **Recargar los daemon antes de realizar cambios**  
   ```bash
   sudo systemctl daemon-reload
   ```

2. **Habilitar y validar el servicio**  
   ```bash
   sudo systemctl enable /home/estudiante/Documents/mlops_taller_mlflow/MLOPS_PUJ/Niveles/2/mlflow/mlflow_serv.service
   sudo systemctl start mlflow_serv.service
   ```

3. **Verificar que el servicio funciona adecuadamente**  
   ```bash
   sudo systemctl status mlflow_serv.service
   ```

---

## 📚 JupyterLab usando Dockerfile

Para aislar la ejecución de la plataforma de la ejecución del código de **Machine Learning**, se creará una imagen de contenedor y se usará para desplegar **Jupyter Notebook**, que contiene ejemplos para el uso de **MLflow**.  

1. **Construir la imagen de JupyterLab**  
   ```bash
   docker build -t jupyterlab .
   ```

2. **Ejecutar el contenedor de JupyterLab**  
   ```bash
   docker run -it --name jupyterlab --rm -e TZ=America/Bogota -p 8888:8888 -v $PWD:/work jupyterlab:latest
   ```

   Accede a la interfaz gráfica en:  
   ```
   http://localhost:8888
   ```

---

🚀 **¡Listo! Ahora puedes gestionar el servicio de MLflow y ejecutar tus experimentos en JupyterLab de forma aislada.**

