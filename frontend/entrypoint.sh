#!/bin/bash

# Check if USE_MTLS is set to 1
if [ "$USE_MTLS" = "1" ]; then
    # Set up certificates
    /app/setup_certs.sh

    # Check if setup was successful
    if [ $? -eq 0 ]; then
        echo "TLS Certificate setup successful, starting the frontend..."
        uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8501
    else
        echo "TLS Certificate setup failed, set USE_MTLS=0 and refer to documentation to set up the server on http"
        exit 1
    fi
else
    # If USE_MTLS is not set to 1, directly start the frontend
    echo "TLS encryption not enabled, to enable set USE_MTLS=1 and refer to documentation"
    uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8501
fi
