FROM python:3.11-slim-bookworm 

# Instala dependencias necesarias para la instalación y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates g++ build-essential 

# Descarga e instala UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Asegura que UV esté en el PATH
ENV PATH="/root/.local/bin:$PATH"

# Crea el entorno virtual en /opt/venv
RUN uv venv /opt/venv

# Usa el entorno virtual automáticamente
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
# Instala paquetes dentro del entorno virtual
RUN uv pip install -r requirements.txt

# Expone el puerto de JupyterLab
EXPOSE 8888

# Comando por defecto para ejecutar JupyterLab sin token
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
