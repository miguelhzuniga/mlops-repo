FROM python:3.12-slim-bookworm

# Establece el directorio de trabajo
WORKDIR /app

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Instala paquetes de ciencia de datos dentro del entorno virtual

RUN uv pip install jupyterlab numpy pandas matplotlib scikit-learn --system

# Expone el puerto de JupyterLab
EXPOSE 8888

# Comando por defecto para ejecutar JupyterLab sin token
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
