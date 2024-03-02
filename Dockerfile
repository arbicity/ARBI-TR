# Start from the NVIDIA CUDA 12.3.1 runtime Ubuntu 22.04 image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for pyenv, Poetry, and your application
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

# Install pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH:/root/.poetry/bin"


RUN curl https://pyenv.run | bash
RUN pyenv install 3.11.4
RUN pyenv global 3.11.4

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$PATH:/root/.local/bin"

# Set the working directory in the container
WORKDIR /app
# Copy the poetry files to the container
COPY pyproject.toml poetry.lock* /app/


# Install Python dependencies
RUN poetry install --no-interaction --no-ansi

# Copy the rest of your application's code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "uvicorn", "whisper-api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

