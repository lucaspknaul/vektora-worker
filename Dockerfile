FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

# Update image and install required packages
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Make sure the venv's pip is used
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside the venv
RUN pip install --upgrade pip

# Install dependencies inside the virtual environment
RUN pip install --no-cache-dir runpod
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu129
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors xformers

WORKDIR /

# Copy your handler file
COPY handler.py /
COPY test_input.json /

# Start the container using the venv Python
CMD ["/opt/venv/bin/python", "-u", "handler.py"]
