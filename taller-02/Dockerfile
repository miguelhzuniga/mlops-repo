#
FROM tensorflow/tfx:1.12.0

WORKDIR /tfx/src/jupyter-notebooks

#Activar anaconda
RUN /bin/bash -c "source activate base"

RUN conda init bash && \
    echo "source ~/.bashrc" >> ~/.bash_profile

RUN conda run -n base python --version

# Instalar dependencias usando anaconda
RUN conda install conda-forge::jupyterlab

# Asignar conda ambiente base en Jupyter 
ENV PATH /opt/conda/bin:$PATH