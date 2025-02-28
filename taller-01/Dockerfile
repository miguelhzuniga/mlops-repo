FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir uv

RUN uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8989

CMD ["uvicorn", "main-app:app", "--host", "0.0.0.0", "--port", "8989"]