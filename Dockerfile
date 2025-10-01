# Python 3.11 slim image
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PySCF & scientific stack
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran \
    openmpi-bin \
    libopenmpi-dev \
    git \
    pkg-config \
    cmake \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
