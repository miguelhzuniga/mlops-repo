FROM apache/airflow:2.10.5-python3.8

ENV AIRFLOW_HOME=/opt/airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt
