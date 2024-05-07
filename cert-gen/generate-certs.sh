#!/bin/bash

# Paths for certificates
CA_CERT="/certs/ca-cert.pem"
CA_KEY="/certs/ca-key.pem"
SERVER_CERT="/certs/server-cert.pem"
SERVER_KEY="/certs/server-key.pem"
CLIENT_CERT="/certs/client-cert.pem"
CLIENT_KEY="/certs/client-key.pem"

echo "Starting certificate generation..."
# Generate CA's private key and certificate without passphrase
openssl req -new -x509 -days 365 -keyout $CA_KEY -out $CA_CERT -nodes \
-subj "/C=GB/ST=London/L=EN/O=ARBI/CN=ARBI-TR-CA"

# Generate server's private key
openssl genrsa -out $SERVER_KEY 2048

# Create server certificate signing request (CSR)
openssl req -new -key $SERVER_KEY -out $SERVER_KEY.req -subj "/C=GB/ST=EN/L=London/O=ARBI/CN=arbi-tr-api-container" \
-config <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=DNS:arbi-tr-api-container, DNS:localhost"))

# Sign server CSR with CA's private key and certificate to generate server certificate
openssl x509 -req -days 365 -in $SERVER_KEY.req -CA $CA_CERT -CAkey $CA_KEY -set_serial 01 -out $SERVER_CERT \
-extensions SAN -extfile <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=DNS:arbi-tr-api-container, DNS:localhost"))

# Generate client's private key
openssl genrsa -out $CLIENT_KEY 2048

# Create client certificate signing request (CSR)
openssl req -new -key $CLIENT_KEY -out $CLIENT_KEY.req -subj "/C=GB/ST=EN/L=London/O=ARBI/CN=arbi-tr-frontend-container" \
-config <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=DNS:arbi-tr-frontend-container, DNS:localhost"))

# Sign client CSR with CA's private key and certificate to generate client certificate
openssl x509 -req -days 365 -in $CLIENT_KEY.req -CA $CA_CERT -CAkey $CA_KEY -set_serial 02 -out $CLIENT_CERT \
-extensions SAN -extfile <(cat /etc/ssl/openssl.cnf <(printf "[SAN]\nsubjectAltName=DNS:arbi-tr-frontend-container, DNS:localhost"))

echo "Certificates generated successfully."
# Clean up CSR files
rm $SERVER_KEY.req $CLIENT_KEY.req
