FROM python:3.9
# Copiar archivos del proyecto
COPY . .

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Instala paquetes de ciencia de datos dentro del entorno virtual

RUN uv pip install fastapi==0.109.1 uvicorn==0.27.0.post1 joblib scikit-learn pandas python-multipart --system


EXPOSE 8888
