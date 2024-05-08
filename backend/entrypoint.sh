#!/bin/bash

# Check if USE_MTLS is set to 1
if [ "$USE_MTLS" = "1" ]; then
    # Run the certificate setup script
    /app/setup_certs.sh

    # Check if setup was successful
    if [ $? -eq 0 ]; then
        echo "TLS Certificate setup successful, starting the server..."
        poetry run python main.py
    else
        echo "TLS Certificate setup failed, use docker-compose.no-tls.yaml if you want to run with http"
        exit 1
    fi
else
    # If USE_MTLS is not set to 1, directly start the server
    echo "WARNING: TLS encryption not enabled, set USE_MTLS=1 and follow documentation to enable encryption"
    poetry run python main.py
fi
