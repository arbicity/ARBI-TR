FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    openssl \
    ffmpeg \
    git \
    libsndfile1 \
    libpython3.12t64 \
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

# Install deps first (cache layer) — install from the committed lock so the
# image matches the tested resolution (reproducible build).
COPY --chown=newuser:newuser pyproject.toml uv.lock /app/
RUN uv sync --no-dev --frozen

# HuggingFace model cache — bind-mounted from host HF_CACHE at runtime
ENV HF_HOME=/app/models

# Application code
COPY --chown=newuser:newuser . /app

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]

ENV DEBIAN_FRONTEND=
