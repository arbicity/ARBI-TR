#!/bin/sh

# Wait for the certificates to be ready
while [ ! -f /app/certs/server-cert.pem ]; do
  echo "Waiting for certificates..."
  sleep 1
done

echo "Certificates found, starting API service..."

# Start your main service here, example with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile=/app/certs/server-key.pem --ssl-certfile=/app/certs/server-cert.pem
