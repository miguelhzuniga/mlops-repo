FROM python:3.9-slim

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Usar la misma estructura de directorios que el proyecto original
WORKDIR /app

# Copiar requisitos primero para aprovechar la caché de Docker
COPY requirements.txt /app/

# Instalar dependencias con pip install y --no-cache-dir
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY main_server.py /app/

# Verificar instalación de mlflow
RUN pip list | grep mlflow

# Exponer el puerto en el que corre la aplicación
EXPOSE 80

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main_server:app", "--host", "0.0.0.0", "--port", "80"]