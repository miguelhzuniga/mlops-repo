FROM ubuntu:latest

RUN apt update
RUN apt install -y wget
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
RUN bash -c "source ~/miniconda3/bin/activate"
RUN bash -c "source ~/miniconda3/bin/activate base && conda init --all"
RUN bash -c "source ~/miniconda3/bin/activate base && pip install joblib fastapi uvicorn"

WORKDIR /app

COPY . .

#CMD [ "ls", "-la", "/app" ]

CMD ["bash", "-c", "source ~/miniconda3/bin/activate base && uvicorn main-app:app --reload --host 0.0.0.0 --port 80"]