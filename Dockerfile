FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y git ninja-build && rm -rf /var/lib/apt/lists/*

# Официальный репозиторий — стабильный, не пропадёт как сторонние копии
RUN git clone https://github.com/TencentARC/InstantMesh.git /workspace/InstantMesh
WORKDIR /workspace/InstantMesh

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Версия xformers, указанная в официальном README именно под torch 2.1.0
RUN pip install --no-cache-dir xformers==0.0.22.post7

# deepspeed требует компиляции CUDA-кода при установке и часто падает в Docker.
# Для генерации моделей (не обучения) он не нужен — отключаем сборку его
# C++/CUDA-расширений переменной окружения, ставится "облегчённый" вариант.
ENV DS_BUILD_OPS=0

# Зависимости самого проекта (файл requirements.txt уже лежит в склонированном репозитории)
RUN pip install --no-cache-dir -r requirements.txt

# runpod — SDK для serverless-воркера, trimesh — для конвертации .obj -> .glb под Unity
RUN pip install --no-cache-dir runpod requests trimesh

COPY handler.py /workspace/InstantMesh/handler.py

CMD ["python", "handler.py"]
