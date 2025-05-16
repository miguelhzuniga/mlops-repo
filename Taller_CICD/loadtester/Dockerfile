FROM python:3.11-slim

WORKDIR /app

COPY locustfile.py .

RUN pip install --no-cache-dir locust

EXPOSE 8089

CMD ["locust", "-f", "locustfile.py", "--host", "http://ml-api-service:8000"]
