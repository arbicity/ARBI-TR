#!/bin/bash

# URL of the server
URL="http://cert-init:8080"

# Ensure the certs directory exists and set permissions
mkdir -p /app/certs
chmod 744 /app/certs

# File paths
CLIENT_KEY="/app/certs/client.key"
CLIENT_CSR="/app/certs/client.csr"
SIGNED_CERT="/app/certs/signed_client.crt"
CA_CERT="/app/certs/ca.crt"

# Retrieve the CA certificate
response=$(curl -s -o $CA_CERT -w "%{http_code}" -X POST "${URL}/getca")

if [ "$response" -eq 200 ]; then
    echo "CA certificate has been retrieved and stored at $CA_CERT"
else
    echo "Failed to retrieve CA certificate. Server response: $response"
    exit 1
fi

# Generate a Private Key if it does not exist
if [ ! -f "$CLIENT_KEY" ]; then
    openssl genrsa -out $CLIENT_KEY 2048
fi

# Generate the CSR with SAN


openssl req -new -key $CLIENT_KEY -out $CLIENT_CSR -subj "/C=US/ST=State/L=City/O=Organization/CN=arbi-tr-frontend-container" -config <(cat /etc/ssl/openssl.cnf <(printf "[req]\ndistinguished_name=req_distinguished_name\nreq_extensions=req_ext\n[req_distinguished_name]\nCN=Common Name\n[req_ext]\nsubjectAltName=@alt_names\n[alt_names]\nDNS.1=arbi-tr-frontend-container\nDNS.2=localhost"))

# Send the CSR and receive the signed certificate
response=$(curl -s -o $SIGNED_CERT -w "%{http_code}" -X POST --data-binary @$CLIENT_CSR "${URL}/sign")

if [ "$response" -eq 200 ]; then
    echo "Signed client certificate has been retrieved and stored at $SIGNED_CERT"
else
    echo "Failed to retrieve signed certificate. Server response: $response"
    exit 1
fi
