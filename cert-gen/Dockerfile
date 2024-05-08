FROM python:3.11-slim

# Install OpenSSL and required packages
RUN apt-get update && apt-get install -y \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Install pyOpenSSL
RUN pip install --no-cache-dir cryptography

# Set the working directory in the container
WORKDIR /app

# Copy the server script and the CA creation script into the container
COPY cert-init.py .
COPY create_ca.sh .

# Make the CA creation script executable and run it to generate the CA key and certificate
RUN chmod +x create_ca.sh && ./create_ca.sh

# Expose port 8080 to be accessible from the host
EXPOSE 8080

# Command to run the server script
CMD ["sh", "-c", "python cert-init.py $CERTS_TO_ISSUE"]
