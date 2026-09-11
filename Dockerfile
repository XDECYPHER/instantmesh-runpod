# =========================================================================
# TripoSR RunPod Serverless Worker
# Базовый образ — CUDA "devel" (не "runtime"!), т.к. torchmcubes собирается
# из исходников и требует nvcc на этапе pip install.
# =========================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# --- Системные зависимости -----------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev git wget \
        libgl1 libglib2.0-0 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

# --- PyTorch (под CUDA 12.1) ----------------------------------------------
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# --- Клонируем официальный репозиторий TripoSR ----------------------------
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /workspace/TripoSR

WORKDIR /workspace/TripoSR

# --- Python-зависимости самого TripoSR ------------------------------------
# torchmcubes ставится отдельно (собирается с CUDA-поддержкой из исходников,
# поэтому нужен образ "devel", а не "runtime")
RUN pip install \
        omegaconf==2.3.0 \
        Pillow==10.1.0 \
        einops==0.7.0 \
        transformers==4.35.0 \
        trimesh==4.0.5 \
        rembg \
        huggingface-hub \
        xatlas \
        onnxruntime \
    && pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git

# --- Наши доп. зависимости под RunPod --------------------------------------
RUN pip install runpod requests

# --- Качаем веса модели ЗАРАНЕЕ (на этапе билда), чтобы не тянуть их       -
# --- при каждом cold start                                                -
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='stabilityai/TripoSR', local_dir='/workspace/model_cache')"

# --- Копируем наш handler --------------------------------------------------
COPY handler.py /workspace/handler.py

WORKDIR /workspace

CMD ["python3", "-u", "handler.py"]
