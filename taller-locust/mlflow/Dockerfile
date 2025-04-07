# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /mlflow

# Copy the local MLFlow files (if you have any) into the container
COPY . /mlflow

# Install system dependencies for MLFlow

RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install MLFlow and any additional Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for AWS S3 or MinIO
# These are necessary for MLflow to interact with your S3-compatible object storage
ENV MLFLOW_S3_ENDPOINT_URL=http://10.43.101.175:9000
ENV AWS_ACCESS_KEY_ID=admin
ENV AWS_SECRET_ACCESS_KEY=supersecret

# Expose the port that MLFlow uses
EXPOSE 5000


# Command to run the MLFlow server
CMD ["mlflow", "server", \
    "--backend-store-uri", "mysql://mlflow:mlflow@mysql/mlflow_db", \
    "--default-artifact-root", "s3://mlflows3/artifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000"]