FROM python:3.9-slim
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
&& rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
 
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy the application code
COPY app/ app/
 
EXPOSE 8501
 
# Run the Gradio app
CMD ["python", "app/app.py"]