# Use phusion/baseimage as base image
FROM phusion/baseimage:jammy-1.0.2

# Set DEBIAN_FRONTEND to noninteractive to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev python3-openssl git tzdata \
    openssl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install pyenv
RUN curl https://pyenv.run | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}"
RUN eval "$(pyenv init --path)" && \
    eval "$(pyenv virtualenv-init -)"

# Install Python 3.11.4 using pyenv
RUN pyenv install 3.11.4 && \
    pyenv global 3.11.4

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

# Copy the certificate setup script and make it executable
COPY setup_certs.sh /app/setup_certs.sh
RUN chmod +x /app/setup_certs.sh

# Copy the rest of your application's code to the container
WORKDIR /app
COPY . /app
RUN poetry install

# Expose the application's port
EXPOSE 8501

# Copy and set the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Command to run the application through the entrypoint script
CMD ["/app/entrypoint.sh"]
