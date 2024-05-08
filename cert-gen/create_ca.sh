#!/bin/bash
# Generate the CA Key and Certificate for signing Client Certs
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -sha256 -days 1825 -out ca.crt -subj "/C=GB/ST=EN/L=London/O=ARBI/CN=CA-INIT"
