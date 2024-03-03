# Start from the NVIDIA CUDA 12.3.1 runtime Ubuntu 22.04 image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for pyenv, Poetry, and your application
# Combine update, install, and cleanup into a single RUN to reduce layers and size
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    ffmpeg \
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

# Install pyenv and Poetry in a single RUN command to utilize cache better and reduce layer size
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH:/root/.poetry/bin"
RUN curl https://pyenv.run | bash \
    && pyenv install 3.11.4 \
    && pyenv global 3.11.4 \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && echo 'export PATH="$PATH:/root/.local/bin"' >> ~/.bashrc

# Set the working directory in the container
WORKDIR /app

# Copy only the files necessary for poetry install to utilize Docker's cache better
# This way, changes in other files won't invalidate the cache for this layer
COPY poetry.lock pyproject.toml /app/

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your application's code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "whisper-api:app", "--host", "0.0.0.0", "--port", "8000"]
