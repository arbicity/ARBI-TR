#!/bin/bash

# Set up certificates
/app/setup_certs.sh

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo "Certificate setup successful, starting the frontend..."
    poetry run streamlit run st-frontend.py --server.address=0.0.0.0 --server.port=8501
else
    echo "Certificate setup failed, terminating..."
    exit 1
fi
