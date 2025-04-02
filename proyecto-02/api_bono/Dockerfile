# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app_bono

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt /app_bono/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 
EXPOSE 8503

# Command to run FastAPI with Uvicorn

#CMD ["uvicorn", "main_bono:app", "--host", "0.0.0.0", "--port", "8503"]
CMD ["python", "gradio_app.py"]