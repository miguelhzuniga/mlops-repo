FROM python:3.8.20

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Usar la misma ruta de trabajo
WORKDIR /app

# Copiar requisitos primero para aprovechar la caché de Docker
COPY requirements.txt /app/

# Instalar dependencias con pip install y --no-cache-dir
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY gradio_app.py /app/

# Verificar instalación de mlflow y gradio
RUN pip list | grep mlflow && pip list | grep gradio

# Exponer el puerto en el que corre la aplicación Gradio
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["python", "/app/gradio_app.py"]