FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

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

RUN useradd -u 10001 -ms /bin/bash newuser \
    && mkdir -p /home/newuser/.local /app \
    && chown -R newuser:newuser /home/newuser /app

USER newuser
WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/newuser/.local/bin:$PATH"

# Install deps first (cache layer)
COPY --chown=newuser:newuser pyproject.toml /app/
# Regenerate lock on build so it stays in sync with pyproject.toml
RUN uv lock && uv sync --no-dev

# HuggingFace model cache — bind-mounted from host HF_CACHE at runtime
ENV HF_HOME=/app/models

# Application code
COPY --chown=newuser:newuser . /app

RUN chmod +x /app/setup_certs.sh /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]

ENV DEBIAN_FRONTEND=
