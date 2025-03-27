FROM python:3.9
COPY ./requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY ./main_server.py /main_server.py
CMD ["uvicorn", "main_server:app", "--host", "0.0.0.0", "--port", "80"]