FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install diffusers transformers accelerate safetensors xformers

WORKDIR /app
COPY main.py .

CMD ["python3", "-m", "runpod.serverless"]
