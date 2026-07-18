FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only version of PyTorch to avoid massive Nvidia CUDA GPU downloads
RUN pip3 install --no-cache-dir --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
