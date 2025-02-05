# Usa última imagen alpine
FROM alpine:latest

# Directorio de tabajo
WORKDIR /data

# Copiar archivo de texto
COPY ex-txt.txt /data/

# Comando por defecto al iniciar el conetendor
#CMD ["echo", "alpine Docker iniciado con éxito"]

#CMD ["echo", "Contenido de la carpeta data"]

CMD ["ls", "-la", "/data"]
