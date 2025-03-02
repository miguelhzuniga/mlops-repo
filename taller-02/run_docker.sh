#!/bin/bash

SERVICE_NAME="jupyter-lab"

cleanup() {
    sudo docker rm -f $(sudo docker ps -aq --filter "name=${SERVICE_NAME}") 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

sudo docker rm -f $(sudo docker ps -aq --filter "name=${SERVICE_NAME}") 2>/dev/null
sudo docker compose build
sudo docker compose up

while true; do sleep 1; done
