#!/bin/bash

# Wait for the CA certificate to be available
while [ ! -f /certs/ca.crt ]
do
  sleep 1
done

# Generate a private key and certificate for the frontend
openssl req -x509 -newkey rsa:4096 -keyout /certs/frontend.key -out /certs/frontend.crt -days 365 -nodes -subj "/CN=localhost"

# Assuming Streamlit needs to securely connect to the backend
export REQUESTS_CA_BUNDLE=/certs/ca.crt

# Start the frontend application with Streamlit
exec poetry run streamlit run st-frontend.py --server.address=0.0.0.0 --server.port=8501
