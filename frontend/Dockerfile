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

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

# Copy the certificate setup script and make it executable
COPY setup_certs.sh /app/setup_certs.sh
RUN chmod +x /app/setup_certs.sh

# Copy the rest of your application's code to the container
WORKDIR /app
COPY . /app
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
RUN poetry install --only main && rm -rf $POETRY_CACHE_DIR

# Expose the application's port
EXPOSE 8501

# Copy and set the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Command to run the application through the entrypoint script
CMD ["/app/entrypoint.sh"]
