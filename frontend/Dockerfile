# Use a base image with Python preinstalled
FROM python:3.11-slim-bookworm

# Set DEBIAN_FRONTEND to noninteractive to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev python3-openssl tzdata \
    openssl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy the certificate setup script and make it executable
COPY setup_certs.sh /app/setup_certs.sh
RUN chmod +x /app/setup_certs.sh

# Set the working directory
WORKDIR /app

# Copy only the files necessary for uv sync first
COPY uv.lock pyproject.toml /app/

# Install dependencies via UV
RUN uv sync --frozen --no-dev

# Using .dockerignore to exclude __pycache__ and secrets.toml
COPY . /app

# Expose the application's port
EXPOSE 8501

# Copy and set the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Command to run the application through the entrypoint script
CMD ["/app/entrypoint.sh"]
