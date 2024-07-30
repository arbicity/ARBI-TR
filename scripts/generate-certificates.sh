#!/bin/sh

# Generate CA key and certificate
openssl req -x509 -newkey rsa:4096 -keyout /certs/ca.key -out /certs/ca.crt -days 365 -nodes -subj '/CN=My CA'

# Generate Server key and certificate
openssl genrsa -out /certs/server.key 4096
openssl req -new -key /certs/server.key -out /certs/server.csr -subj '/CN=server.local'
openssl x509 -req -in /certs/server.csr -CA /certs/ca.crt -CAkey /certs/ca.key -CAcreateserial -out /certs/server.crt -days 365

# Generate Client key and certificate
openssl genrsa -out /certs/client.key 4096
openssl req -new -key /certs/client.key -out /certs/client.csr -subj '/CN=client.local'
openssl x509 -req -in /certs/client.csr -CA /certs/ca.crt -CAkey /certs/ca.key -CAcreateserial -out /certs/client.crt -days 365

# Cleanup
rm /certs/*.csr
chmod -R 644 /certs/*
