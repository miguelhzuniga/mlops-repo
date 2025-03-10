# Taller de MLOps con Airflow y Docker Compose

Este repositorio contiene una solución para un taller de MLOps utilizando Docker Compose, Apache Airflow y una base de datos MySQL. Se implementa un flujo de trabajo que permite la carga, preprocesamiento, entrenamiento de modelos y la creación de una API para realizar inferencias.

## 📌 Requisitos

- Docker y Docker Compose instalados
- Git instalado

## 🚀 Instalación y Configuración

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/miguelhzuniga/mlops-repo.git
   cd mlops-repo/taller_airflow
   ```

2. Iniciar los servicios con Docker Compose:

   ```bash
   docker-compose up -d
   ```

3. Acceder a la interfaz de Airflow:

   - URL: [http://localhost:8080](http://localhost:8080)
   - Usuario y contraseña: Definidos en el archivo `docker-compose.yml`

## 🏗️ Arquitectura del Proyecto

El sistema se compone de los siguientes servicios:

- **MySQL**: Base de datos para almacenar los datos de Penguins y los datos preprocesados.
- **Airflow**: Orquestador de tareas que gestiona el flujo de datos y el entrenamiento del modelo.
- **API de Inferencia**: Servicio que expone el modelo entrenado para realizar predicciones.

## 📜 DAGs Implementados

Se han definido los siguientes DAGs en Airflow:

1. **Borrar contenido de la base de datos**
2. **Cargar datos de Penguins sin preprocesar**
3. **Preprocesar datos para entrenamiento**
4. **Entrenar modelo de Machine Learning**
5. **Implementar API de inferencia**

## 📊 Uso de la API de Inferencia

Una vez desplegados los servicios, la API estará disponible en `http://localhost:8888/docs`. 

Ejemplo de uso en bash:

1. Crear id de item 1
```bash
curl "http://localhost:8888/get-item/1"
```
2. Enviar datos para predecir al item 1
```bash
curl -X POST "http://localhost:8888/items/1" -H "Content-Type: application/json" -d '{
  "item": {
    "island": "Torgersen",
    "culmen_length_mm": 10,
    "culmen_depth_mm": 10,
    "flipper_length_mm": 10,
    "body_mass_g": 10,
    "sex": "MALE",
    "species": "string"
  },
  "modelo": {
    "modelo": "LogRegCV"
  }
}'
```
3. Retornará la especie en "species":
```bash
[{"island":"Torgersen","culmen_length_mm":10.0,"culmen_depth_mm":10.0,"flipper_length_mm":10.0,"body_mass_g":10.0,"sex":"MALE","species":"Adelie"},{"modelo":"LogRegCV"}]
```

## 🛑 Detener los Servicios

Para detener y eliminar los contenedores, ejecutar:

```bash
docker-compose down
```

## 📝 Contribuciones

- Luis Frontuso
- Camilo Serrano
- Miguel Zúñiga