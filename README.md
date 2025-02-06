ARBI TR is a self-hosted, open-source, production-grade audio transcription tool using Whisper speech recognition and Pyannote's diarization capabilities with a user-friendly self-service interface.

ARBI TR provides up to 400x realtime transcription speeds on our ARBI BOX device powered by a single NVIDIA RTX 4090 GPU.  Than means a 1 hour video can be transcribed in just 9 seconds.

### Key Features:
- **Self-hosted Solution:** Ability to self-host provides data privacy, control, and cost savings vs cloud-based APIs.
- **Containerized frontend/backend architecture:** Modular frontend/backend architecture with Docker simplifies deployment.
- **FastAPI Server:** Offers a robust GPU-accelerated API backend for scalable deployment.
- **Advanced Transformers:** Utilizes the latest Huggingface Transformers library features, including SDPA and Flash Attention 2, for ultra-fast speech recognition.
- **Whisper Integration:** Uses Whisper v3-large by default for state-of-the-art speech recognition.
- **Speechbrain and Pyannote for Diarization:** Uses Speechbrain voice embedding models with a custom speaker identification and diarization pipeline.
- **Streamlit Frontend:** Provides a simple intuitive user experience.
- **TLS Encryption:** Provides custom solution for TLS encryption between docker containers.
- **Kubernetes-ready:**  Docker images can be used as-is in a Kubernetes production environment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Prerequisites

Before you start, ensure you have the following installed:
- Docker
- NVIDIA Container Toolkit (for running with Docker)
- Git (for cloning the repository)

For running without Docker and for development purposes, you will need:
- pyenv
- Python 3.11.4
- Poetry
- NVIDIA CUDA 12.3.1 (other versions might work depending on your setup)

## Getting Started with Docker

The easiest way to launch ARBI TR is by using Docker. This method automatically sets up the environment and starts the application with minimal setup.

1. Clone the repository:
    ```bash
    git clone https://github.com/arbitrationcity/ARBI-TR.git
    ```
2. Navigate to the cloned directory:
    ```bash
    cd ARBI-TR
    ```
3. Launch the application using Docker Compose:
    ```bash
    docker compose up
    ```

This will build and start the containers, serving the frontend at http://localhost:8501 and the backend as a FASTAPI endpoint at http://localhost:8000/transcribe/ (see http://localhost:8000/docs for more endpoint information).

## Running Without Docker (For Development)

If you wish to contribute to the project or just prefer to run the application without Docker, follow these steps:

### Setting Up the Backend

1. Install `pyenv` and set up Python 3.11.4:
    ```bash
    pyenv install 3.11.4
    pyenv global 3.11.4
    ```
2. Install Poetry:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
3. Navigate to the backend directory and set up the environment:
    ```bash
    cd backend
    poetry shell
    poetry install
    ```
4. Start the FastAPI server:
    ```bash
    uvicorn whisper-api:app --host 0.0.0.0 --port 8000
    ```

### Setting Up the Frontend

1. Ensure you are in the root directory of the project, then change to the frontend directory:
    ```bash
    cd frontend
    ```
2. Set up the environment:
    ```bash
    poetry shell
    poetry install
    ```
3. Start the Streamlit frontend:
    ```bash
    API_ENDPOINT="http://localhost:8000/transcribe/" STREAMLIT_SERVER_MAX_UPLOAD_SIZE="500" streamlit run st-frontend.py --server.port 8501 --server.address 0.0.0.0
    ```

## Using ARBI TR

After starting ARBI TR using either Docker or the development setup, open your web browser and navigate to `http://localhost:8501` to access the Streamlit frontend. From here, you can upload video or audio files for transcription.

For more detailed usage instructions and troubleshooting, please refer to the documentation in the `docs` folder.

## Contributing

Contributions to ARBI TR are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## Acknowledgments

This project relies on openai/whisper (speech recognition model), speechbrain/spkrec-ecapa-voxceleb (speaker embeddings model), pyannote/pyannote-audio (diarization pipeline), and huggingface/transformers (ultra-fast ASR pipeline), and draws inspiration from vumichien/Whisper_speaker_diarization among other projects. Special thanks to all contributors and maintainers for making ARBI TR possible.
