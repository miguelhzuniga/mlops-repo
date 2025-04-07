FROM jupyter/base-notebook:python-3.10

# Copy the requirements.txt into the container
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

