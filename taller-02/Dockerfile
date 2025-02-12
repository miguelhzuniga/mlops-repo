FROM python:3.9
# Copiar archivos del proyecto
COPY . .
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
CMD ["uvicorn", "main-app:app", "--host", "0.0.0.0", "--port", "80"]
