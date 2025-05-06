# Usar la imagen base de Jupyter con Python 3.10
FROM jupyter/base-notebook:python-3.10

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt /tmp/requirements.txt

# Copiar el archivo experimentos.ipynb al contenedor
COPY experimentos.ipynb /home/jovyan/experimentos.ipynb

# Cambiar temporalmente a root para poder cambiar los permisos
USER root

# Asegurar permisos correctos en el archivo experimentos.ipynb
RUN chmod 755 /home/jovyan/experimentos.ipynb

# Volver al usuario jovyan
USER jovyan

# Instalar las dependencias de Python desde requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Exponer el puerto 8888 para que JupyterLab pueda ser accesible desde el navegador
EXPOSE 8888

# Definir el comando para iniciar JupyterLab sin token
CMD ["start-notebook.sh", "--NotebookApp.token=''"]
