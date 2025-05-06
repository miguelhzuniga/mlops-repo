FROM apache/airflow:2.10.5-python3.8

ENV AIRFLOW_HOME=/opt/airflow

# Switch to root to install system packages
USER root

# Install libgomp1 (needed for OpenMP support used by LightGBM, XGBoost, etc.)
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
