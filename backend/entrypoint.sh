#!/bin/bash

# Run the certificate setup script
/app/setup_certs.sh

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo "Certificate setup successful, starting the server..."
    poetry run python main.py
else
    echo "Certificate setup failed, terminating..."
    exit 1
fi
