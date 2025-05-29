FROM apache/airflow:2.6.0-python3.8

USER root

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgcc-s1 \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    apache-airflow-providers-postgres \
    mlflow==2.10.0 \
    evidently \
    scikit-learn==1.0.2 \
    pandas \
    numpy==1.21.0 \
    lightgbm \
    boto3 \
    psycopg2-binary==2.9.7 \
    requests==2.31.0 \
    joblib \
    matplotlib \
    seaborn \
    plotly \
    httpx \
    aiohttp \
    python-dateutil \ 
    pydantic==1.10.12 \
    pyarrow==10.0.1