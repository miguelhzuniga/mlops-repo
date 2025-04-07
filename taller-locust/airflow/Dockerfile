# Use the official Apache Airflow image as a base image
FROM apache/airflow:2.10.5-python3.8

# Set environment variables (optional but good practice)
ENV AIRFLOW_HOME=/opt/airflow

# Copy the requirements.txt file into the container
COPY requirements.txt /requirements.txt

# Install the Python dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r /requirements.txt

