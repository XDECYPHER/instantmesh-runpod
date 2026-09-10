# Базовый образ с уже установленными PyTorch 2.1.0 + CUDA 11.8 и nvcc
# (nvcc обязателен — без него не скомпилируется torchmcubes)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Клонируем официальный репозиторий TripoSR
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /app/TripoSR

WORKDIR /app/TripoSR

# Ставим зависимости самого TripoSR
RUN pip install --upgrade setuptools pip
RUN pip install -r requirements.txt

# torchmcubes ставим отдельно из исходников — так надёжнее,
# чем готовое pip-колесо, которое часто не совпадает по версии CUDA
RUN pip uninstall -y torchmcubes || true
RUN pip install git+https://github.com/tatsy/torchmcubes.git

# Библиотека для работы как RunPod Serverless воркер
RUN pip install runpod

# Наш обработчик запросов
COPY handler.py /app/TripoSR/handler.py

CMD ["python", "-u", "handler.py"]
