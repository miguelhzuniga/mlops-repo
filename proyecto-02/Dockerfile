FROM python:3.9
ARG MLFLOW_S3_ENDPOINT_URL=http://10.43.101.202.109:9000
ARG AWS_ACCESS_KEY_ID=admin
ARG AWS_SECRET_ACCESS_KEY=supersecret
RUN mkdir /work
WORKDIR /work
COPY . .
RUN pip install jupyter==1.0.0 -U && pip install jupyterlab==3.6.1
RUN pip install -r requirements.txt
EXPOSE 8888

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN pip install --no-cache-dir fastapi uvicorn==0.27.0.post1 mlflow==2.3.0 numpy==1.24.4 psutil==7.0.0 scipy==1.11.4 pandas pydantic scikit-learn==1.2.2 dill python-multipart requests==2.28.2 boto3==1.26.121
RUN uvicorn app.main-app:app --host 0.0.0.0 --port 8000


ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]