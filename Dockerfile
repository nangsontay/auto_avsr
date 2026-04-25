FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# Install FFmpeg 7.x from jellyfin repo — Ubuntu 22.04 apt ships FFmpeg 4.x
# which lacks Blackwell (RTX 5090) NVENC support
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg && \
    curl -fsSL https://repo.jellyfin.org/ubuntu/jellyfin_team.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/jellyfin.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/jellyfin.gpg] https://repo.jellyfin.org/ubuntu jammy main" \
      > /etc/apt/sources.list.d/jellyfin.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git wget jellyfin-ffmpeg7 \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1-mesa-glx \
    libsndfile1 libportaudio2 \
    python3.10 python3.10-dev python3-pip \
    gcc \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg \
    && ln -sf /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the pip section of environment.yml as requirements.txt (layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/hhj1897/face_detection.git /tmp/face_detection && \
    pip install -e /tmp/face_detection

RUN git clone https://github.com/hhj1897/face_alignment.git /tmp/face_alignment && \
    pip install -e /tmp/face_alignment

# Copy project code
COPY . .