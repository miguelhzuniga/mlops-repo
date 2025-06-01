FROM apache/airflow:2.10.5-python3.8

ENV AIRFLOW_HOME=/opt/airflow

USER root

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
