# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container to /app
WORKDIR /app

# Install dependencies & python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    ffmpeg \
    gcc \
    g++ \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && apt update && apt install -y python3.10 python3-pip

# Clone the GitHub repository
RUN git clone https://github.com/daswer123/FollowYourEmoji-Webui .

# Run install script
RUN chmod +x install_no_venv.sh && ./install_no_venv.sh

RUN pip install --upgrade pip

# Expose port 7860 for the gradio app
EXPOSE 7860

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN chmod +x start.sh

# Run start.sh when the container starts
CMD ["bash","start.sh"]
