#!/bin/bash

# Wait for the CA certificate to be available
while [ ! -f /certs/ca.crt ]; do
  sleep 1
done

# Activate Poetry virtual environment
source $HOME/.bashrc
poetry shell

# Generate a private key and certificate for the backend
openssl req -x509 -newkey rsa:4096 -keyout /certs/backend.key -out /certs/backend.crt -days 365 -nodes -subj "/CN=localhost"

# Start the backend using Uvicorn with SSL
exec uvicorn whisper-api:app --host 0.0.0.0 --port 8000 --ssl-keyfile=/certs/backend.key --ssl-certfile=/certs/backend.crt --ssl-ca-certs=/certs/ca.crt
