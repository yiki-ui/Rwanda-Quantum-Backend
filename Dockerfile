# Python 3.10 
FROM python:3.11.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for pyscf and other scientific libraries
# These are common build tools and libraries for scientific Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran \
    openmpi-bin \
    libopenmpi-dev \
    git \
    pkg-config \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Expose the port your FastAPI application runs on
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
