# README para el Clúster de Airflow con CeleryExecutor, Redis y PostgreSQL

Este repositorio configura un entorno de desarrollo local para ejecutar Apache Airflow con CeleryExecutor, Redis como broker de tareas y PostgreSQL como backend de la base de datos. Además, el entorno incluye MinIO para almacenamiento de artefactos, MySQL para almacenar metadata de MLFlow, PgAdmin para monitorear la base de datos de airflow y JupyterLab como ambiente de desarrollo.

---

## Servicios en Docker Compose

El entorno de desarrollo está compuesto por varios contenedores Docker que incluyen Airflow, bases de datos, almacenamiento de artefactos y más. A continuación, se describen los servicios principales:

### 1. **Airflow**: Clúster con CeleryExecutor

Airflow se ejecuta utilizando CeleryExecutor, y se conecta a Redis para la gestión de tareas y PostgreSQL para el almacenamiento de metadatos.

- **Airflow Webserver**: Accede a la UI de Airflow en `http://localhost:8080`
- **Airflow Scheduler**: Controla la ejecución de tareas programadas.
- **Airflow Worker**: Ejecuta las tareas programadas por el scheduler.
- **Airflow Triggerer**: Gestiona trabajos que activan tareas manualmente.

  ![alt text](images/captura_airflow.png)

### 2. **Redis**: Broker de Celery

Redis se usa como broker para las tareas de Celery. Es necesario para la comunicación entre el `Scheduler` y los `Workers` de Airflow.

### 3. **PostgreSQL**: Backend de Airflow

PostgreSQL se usa como base de datos para almacenar los metadatos de Airflow. Se conecta con los `Scheduler` y `Workers`.

### 4. **MinIO**: Almacenamiento S3

MinIO emula un almacenamiento tipo S3, utilizado para almacenar los artefactos de MLFlow.

- Accede a la consola web de MinIO en `http://localhost:9001` utilizando las credenciales:
  - Usuario: `admin`
  - Contraseña: `supersecret`

- Se debe crear el bucket para almacenar los artefactos de MLFLOW en la interfaz gráfica por medio del botón "Create bucket" y este debe llamarse mlflows3 para que mlflow pueda reconocerlo y se guarde la información.

![alt text](images/captura_minio.png)

### 5. **MLFlow**: Plataforma de Gestión de Experimentos

MLFlow se utiliza para el seguimiento de experimentos de Machine Learning. Se conecta a MySQL para almacenar metadatos.

- Accede a la interfaz web de MLFlow en `http://localhost:5000`.

![alt text](images/captura_mlflow.png)

### 6. **PgAdmin**: Interfaz de Administración de PostgreSQL

PgAdmin proporciona una interfaz gráfica para gestionar la base de datos de PostgreSQL.

- Accede a la interfaz web de PgAdmin en `http://localhost:5050`.
  - Usuario: `admin@example.com`
  - Contraseña: `admin`
Se debe registrar la base de datos con:
- hostname: postgres
- user: airflow
- password: airflow

Para esto se da click en "Servers">"Register" y se ingresan los datos anteriores. Luego se da "Save". Aparecerá la base de datos a la cual si se despliega se podrá dar click derecho y poner "Query tools" y luego hacer consultas como por ejemplo "select * from covertype".

De este modo se puede comprobar que los datos que airflow genera con el dag Cargar_datos.py están cargados a la base de datos de Postgresql

![alt text](images/captura_pgadmin.png)

### 7. **MySQL**: Base de Datos de MLFlow

MySQL se usa para almacenar los metadatos de MLFlow. Se conecta con el contenedor de MLFlow.

### 8. **JupyterLab**: Entorno de Desarrollo

JupyterLab proporciona un entorno interactivo para desarrollo y pruebas. Puedes acceder a él en `http://localhost:8888` usando el token `devtoken`.

![alt text](images/captura_jupyter.png)

### 9. **FastAPI (API)**: API de Inferencia

FastAPI proporciona un servidor para exponer endpoints que interactúan con otros servicios como Airflow y MLFlow. Accede al servidor en `http://localhost:8000/docs`.

Aqui se puede utilizar el mejor modelo generado tras los experimentos de MLFLOW para realizar inferencia.

* Nota: el modelo debe estar en estado de producción en Mlflow y debe llamarse modelo1.

![alt text](images/captura_api.png)

### 10. **FastAPI (API Server)**: Servidor de API

Esta API simula una URL de internet que provee un batch de datos cada 5 segundos, para un total de 10 batches, los cuales serán utilizados para entrenar posteriormente el modelo en airflow y registrar los experimentos y el mejor modelo en mlflow. Los datos se almacenan en la base de datos "airflow" dentro de la tabla "covertype".
`http://localhost:80`.

---

## Uso

Para ejecutar los contenedores y configurar el entorno, sigue estos pasos:

IP MV: 10.43.101.202


1. **Construir los contenedores**:

   En el directorio raíz del proyecto, ejecuta:

   ```bash
   docker-compose up -d --build
   ```
    Para ver el estatus de los servicios:
   ```bash
   docker-compose ps
   ```
    Para bajar los servicios:
   ```bash
   docker-compose down
   ```

   Para bajar los servicios limpiando todo:
   ```bash
   docker-compose down --volumes --remove-orphans
   ```

2. **Prender dags programados**:

    En airflow existen 4 dags que se ejecutan diariamente desde el 30 de marzo del 2025:
    1. Borrar_datos: limpia la base de datos si existe. Se ejecuta posterior a la generación del modelo y está programado 1 hora después de el primer dag. 
    2. Cargar_datos: trae los datos consolidados de la api server tarda aproximadamente 50 segundos y está programado a las 0 horas.
    3. Entrenamiento_mode: entrena un RandomForest con los datos cargados, el cual está programado 2 minutos después de cargar los datos.
    4. Procesa_data: genera experimentos para encontrar los mejores hiperparámetros y por ende el mejor modelo que será utilizado posteriormente por el usuario. Se ejecuta 10 minutos después de cargar los datos.


    * La ejecución recomendada para efectuar el proceso por completo es ejecutar los dags en el orden en el que están enumerados: primero se limpia la base de datos si existe, se cargan los datos, se entrena el modelo y se genera experimentos.

3. **Inferencia en API - OPCIÓN GRADIO**:
    Tras la ejecución de los dags, el usuario ingresará a `http://localhost:8503/`, donde podrá hacer inferencia con el modelo como el siguiente ejemplo:

    1. Entrar a la API. Si se da click en botón "Mapear modelos" y no se ha cargado el modelo aparecerá el mensaje:

![alt text](images/gradio1.png)

    2. Se debe haber puesto el modelo en etapa de Producción antes
![alt text](images/captura_interfaz1.5.png)

    3. Luego se debe dar click al boton 'Mapear modelos' 
![alt text](images/gradio2.png)

    4. Posteriormente se selecciona del dropdown el modelo y se da click en el boton 'Cargar modelo' donde saldra tras unos segundos un aviso con la frase 'modelo seleccionado y cargado correctamente'

![alt text](images/gradio3.png)

    5. Se va a la pestaña 'Realizar predicción', se ingresan los datos de las variables independientes (si se requiere información se puede ir a la pestaña de 'Información'):

![alt text](images/gradio4.png)

![alt text](images/gradio4.5.png)

    6. Se da click en botón "Realizar predicción". Aquí se puede ver el resultado y la predicción dada por el modelo:

![alt text](images/gradio5.png)

4. **Inferencia en API - OPCIÓN DESARROLLO WEB**:
    Tras la ejecución de los dags, el usuario ingresará a `http://localhost:8000/`, donde podrá hacer inferencia con el modelo como el siguiente ejemplo:

  1. Entrar a la API. Si se da click en botón "Mapear modelos" y no se ha cargado el modelo aparecerá el mensaje:

![alt text](images/captura_interfaz1.png)

  2. Se debe haber puesto el modelo en etapa de Producción antes:

![alt text](images/captura_interfaz1.5.png)

  3. Tras poner el modelo en etapa de Producción se da click en "Mapear modelos" y se da click sobre el modelo. Abarecerá la frase: "Modelo seleccionado correctamente"

![alt text](images/captura_interfaz2.png)

  4. Se ingresan los datos de las variables independientes:

![alt text](images/captura_interfaz3.png)

  5. Se da click en botón "Realizar predicción". Aquí se puede ver el resultado y la predicción dada por el modelo:

![alt text](images/captura_interfaz4.png)


**Importante**
  * Para poder almacenar los modelos y experimentos de MLFLOW se debe haber creado el bucket manualmente en MINIO con el nombre mlflows3, al no hacerlo no se registrara la informacion en MLFLOW. 
  * Para poder hacer inferencia de la API por primera vez se debe realizar todo el proceso de ejecucion de los dags debido a que la api requiere del modelo que se genera en el dag 3. Si esto no se aplica entonces la API no se habilitara.

**Autores:**

* Luis Frontuso
* Miguel Zuñiga
* Camilo Serrano
