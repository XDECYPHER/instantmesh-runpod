# =========================================================================
# Stable Fast 3D (SF3D) RunPod Serverless Worker
# Базовый образ — CUDA "devel", т.к. texture_baker и uv_unwrapper —
# CUDA-расширения, которые компилируются из исходников пори pip install.
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

# --- PyTorch (под CUDA 12.1, та же связка, что уже проверена у нас) ------
RUN pip install --upgrade pip && \
    pip install -U setuptools==69.5.1 wheel && \
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# --- Клонируем официальный репозиторий Stable Fast 3D ---------------------
RUN git clone --depth 1 https://github.com/Stability-AI/stable-fast-3d.git /workspace/stable-fast-3d

WORKDIR /workspace/stable-fast-3d

# --- Фикс известного бага: gpytoolbox==0.2.0 не имеет готовых wheel ------
RUN sed -i 's/gpytoolbox==0.2.0/gpytoolbox==0.3.3/' requirements.txt

# --- Ставим основные python-зависимости (без локальных CUDA-пакетов) ----
RUN grep -v -E '^\./' requirements.txt > requirements_main.txt \
    && pip install -r requirements_main.txt

# --- texture_baker и uv_unwrapper: CUDA-расширения, собираем ПОСЛЕ torch,
# --- обязательно с --no-build-isolation (иначе pip не видит torch) -------
RUN pip install --no-build-isolation ./texture_baker/ ./uv_unwrapper/

# --- Наши доп. зависимости под RunPod --------------------------------------
RUN pip install runpod requests

# --- Качаем веса модели ЗАРАНЕЕ (на этапе билда) ---------------------------
# Модель ЗАКРЫТА (gated) на HF — нужен токен с ранее данным доступом.
# Передаём его как build-arg (см. GitHub Actions workflow ниже),
# но НЕ оставляем в финальных слоях образа как секрет.
ARG HF_TOKEN
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='stabilityai/stable-fast-3d', local_dir='/workspace/model_cache', token='${HF_TOKEN}')"

# --- Копируем наш handler --------------------------------------------------
COPY handler.py /workspace/handler.py

WORKDIR /workspace

CMD ["python3", "-u", "handler.py"]
