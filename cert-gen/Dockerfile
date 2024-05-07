# Use a specific version of Ubuntu to ensure consistent environments
FROM ubuntu:22.04

# Set a non-interactive shell environment to prevent issues during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    openssl \
    cron \
    && rm -rf /var/lib/apt/lists/*  # Clean up to keep the image small

# Create a separate directory for scripts
WORKDIR /scripts

# Copy the certificate generation script into the container
COPY generate-certs.sh /scripts/generate-certs.sh

# Ensure the script is executable
RUN chmod +x /scripts/generate-certs.sh

# Create a cron file to schedule your script to run at midnight every day
# Ensure there is a newline at the end of this command
RUN echo "0 0 * * * /scripts/generate-certs.sh >> /var/log/cron.log 2>&1" > /etc/cron.d/cert-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/cert-cron

# Apply the cron job
RUN crontab /etc/cron.d/cert-cron

# Run the script immediately when the container starts, then start cron to handle periodic execution
CMD /scripts/generate-certs.sh && cron -f
