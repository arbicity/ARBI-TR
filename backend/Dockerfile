FROM nvcr.io/nvidia/pytorch:24.04-py3

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for your application
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
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

RUN curl https://pyenv.run | bash \
    && pyenv install 3.11.4 \
    && pyenv global 3.11.4

# Install Poetry using the new recommended installer
RUN curl -sSL https://install.python-poetry.org | python3 -

# Explicitly set the PATH to include Poetry
ENV PATH="/root/.local/bin:$PATH"

# Confirm poetry installation and location
RUN which poetry

# Set the working directory in the container
WORKDIR /app

# Copy only the files necessary for poetry install to utilize Docker's cache better
COPY poetry.lock pyproject.toml /app/

# Install Python dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your application's code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["poetry", "run", "python", "main.py"]
