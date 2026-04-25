FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# Install build deps + runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl gnupg \
    build-essential nasm yasm pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1-mesa-glx \
    libsndfile1 libportaudio2 \
    libx264-dev libx265-dev \
    python3.10 python3.10-dev python3-pip \
    gcc \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && rm -rf /var/lib/apt/lists/*

# nv-codec-headers n13.0.19.0 — first release with Blackwell (SM 12.0) NVENC support
RUN git clone --depth 1 --branch n13.0.19.0 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && make -C /tmp/nv-codec-headers install \
    && rm -rf /tmp/nv-codec-headers

# Build FFmpeg 7.1 from source against the new nv-codec-headers
RUN git clone --depth 1 --branch n7.1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg && \
    cd /tmp/ffmpeg && \
    ./configure \
      --prefix=/usr/local \
      --enable-shared --disable-static --disable-doc \
      --enable-gpl --enable-version3 --enable-nonfree \
      --enable-libx264 --enable-libx265 \
      --enable-cuvid --enable-nvenc --enable-nvdec \
      --enable-ffnvcodec \
      --extra-cflags="-I/usr/local/cuda/include" \
      --extra-ldflags="-L/usr/local/cuda/lib64" \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/ffmpeg
    
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