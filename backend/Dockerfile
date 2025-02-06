FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for your application
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    openssl \
    ffmpeg \
    git \
    libsndfile1 \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    xz-utils \
    make \
    python3-openssl \
    && rm -rf /var/lib/apt/lists/*

# Add newuser and set initial permissions, ensure /app is owned by newuser
RUN useradd -u 10001 -ms /bin/bash newuser \
    && mkdir -p /home/newuser/.local /app \
    && chown -R newuser:newuser /home/newuser /app

# Install pyenv as newuser
USER newuser
ENV PYENV_ROOT="/home/newuser/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

# Switch to newuser before working in the /app directory
WORKDIR /app

# Install pyenv and set up Python as newuser
RUN curl https://pyenv.run | bash \
    && pyenv install 3.11.4 \
    && pyenv global 3.11.4

# Install Poetry using the newuser
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/home/newuser/.local/bin:$PATH"

# Confirm poetry installation and location
RUN which poetry

# Copy only the files necessary for poetry install to utilize Docker's cache better
COPY --chown=newuser:newuser poetry.lock pyproject.toml /app/

ENV POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install Python dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-ansi --no-root --only main && rm -rf $POETRY_CACHE_DIR

# Copy the rest of your application's code to the container
COPY --chown=newuser:newuser . /app

# Copy and setup certificate handling scripts
COPY --chown=newuser:newuser setup_certs.sh /app/
RUN chmod +x /app/setup_certs.sh

# Expose the port the app runs on
EXPOSE 8000

# Copy and set the entrypoint script
COPY --chown=newuser:newuser entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh


# Set the entrypoint script as the command to run
CMD ["/app/entrypoint.sh"]

# Reset DEBIAN_FRONTEND environment variable
ENV DEBIAN_FRONTEND=
