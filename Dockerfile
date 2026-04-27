# Base: latest CUDA 12.8 + cuDNN 9, khớp cu128 wheel của PyTorch mới nhất
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# Bypass cuDNN 9 frontend "FIND" path.
# cuDNN 9.5+ đã loại bỏ một số precompiled engine cho Volta (sm_70 / V100),
# khiến `torch.jit.trace` của FANPredictor lỗi "FIND was unable to find an engine".
# Env này ép cuDNN dùng v7-style algorithm finder (vẫn có engine cho Volta).
# Vô hại trên Ampere/Hopper/Ada/Blackwell.
ENV TORCH_CUDNN_V8_API_DISABLED=1

# Install build deps + runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl gnupg ca-certificates \
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

# nv-codec-headers tương thích CUDA 12.8 SDK
RUN git clone --depth 1 --branch n13.0.19.0 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && make -C /tmp/nv-codec-headers install \
    && rm -rf /tmp/nv-codec-headers

# Build FFmpeg 7.1.1 từ source (có NVENC/NVDEC, x264, x265).
# Chọn 7.x vì torchcodec stable đang ship libtorchcodec_core7.so.
RUN git clone --depth 1 --branch n7.1.1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg \
    && cd /tmp/ffmpeg \
    && ./configure \
        --prefix=/usr/local \
        --enable-shared --disable-static --disable-doc \
        --enable-gpl --enable-version3 --enable-nonfree \
        --enable-libx264 --enable-libx265 \
        --enable-cuvid --enable-nvenc --enable-nvdec \
        --enable-ffnvcodec \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/ffmpeg

WORKDIR /app

# Cài PyTorch stack từ cu128 wheel index TRƯỚC các deps khác,
# để torch + torchvision + torchaudio + torchcodec luôn khớp ABI
# và pip resolver không "downgrade ngược" khi cài requirements.txt.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio torchcodec

# Các Python deps còn lại (đã loại bỏ torch* và triton khỏi requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# face_detection + face_alignment (Imperial College, hhj1897)
RUN git clone https://github.com/hhj1897/face_detection.git /tmp/face_detection \
    && pip install -e /tmp/face_detection

RUN git clone https://github.com/hhj1897/face_alignment.git /tmp/face_alignment \
    && pip install -e /tmp/face_alignment

# Copy project code
COPY . .
