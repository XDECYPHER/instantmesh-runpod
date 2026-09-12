"""
RunPod Serverless handler для Stable Fast 3D (SF3D) (image -> 3D mesh).

Ожидаемый вход (event["input"]):
{
    "image_base64": "<base64 картинки>",   # либо
    "image_url": "https://...",            # одно из двух обязательно
    "foreground_ratio": 0.85,              # опционально
    "texture_resolution": 1024,            # опционально, разрешение текстуры
    "remesh_option": "none",               # none | triangle | quad
    "target_vertex_count": -1,             # -1 = без уменьшения полигонов
    "output_format": "glb"                 # SF3D нативно экспортирует glb
}

Выход:
{
    "mesh_base64": "...",       # если S3/bucket не настроен
    "mesh_url": "...",          # если настроен bucket через env RunPod
    "format": "glb"
}
"""

import os
import sys
import io
import base64
import traceback
import tempfile
from contextlib import nullcontext

import runpod
import torch
import rembg
import requests
from PIL import Image

sys.path.append("/workspace/stable-fast-3d")

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground

DEVICE = get_device() if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/workspace/model_cache"

# =============================================================================
# Модель и rembg-сессия грузятся ОДИН РАЗ при старте воркера, не внутри
# handler(), чтобы не платить за загрузку весов на каждый запрос.
# =============================================================================
print(f"[SF3D] Загружаю модель на устройство: {DEVICE}")
model = SF3D.from_pretrained(
    MODEL_DIR,
    config_name="config.yaml",
    weight_name="model.safetensors",
)
model.to(DEVICE)
model.eval()
rembg_session = rembg.new_session()
print("[SF3D] Модель загружена и готова к работе.")


def _load_image(job_input: dict) -> Image.Image:
    """Достаём картинку из base64 либо по URL."""
    if job_input.get("image_base64"):
        raw = base64.b64decode(job_input["image_base64"])
        return Image.open(io.BytesIO(raw)).convert("RGBA")

    if job_input.get("image_url"):
        resp = requests.get(job_input["image_url"], timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")

    raise ValueError("Нужно передать 'image_base64' или 'image_url' во входных данных.")


def _upload_or_encode(file_path: str) -> dict:
    """Заливка в bucket (если настроен) либо base64 в самом ответе."""
    try:
        from runpod.serverless.utils import rp_upload

        if os.environ.get("BUCKET_ENDPOINT_URL"):
            url = rp_upload.upload_file_to_bucket(
                file_name=os.path.basename(file_path),
                file_location=file_path,
            )
            return {"mesh_url": url}
    except Exception as e:
        print(f"[SF3D] Не удалось загрузить в bucket, fallback на base64: {e}")

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return {"mesh_base64": encoded}


def handler(event):
    try:
        job_input = event.get("input", {})

        image = _load_image(job_input)
        image = remove_background(image, rembg_session)

        foreground_ratio = float(job_input.get("foreground_ratio", 0.85))
        image = resize_foreground(image, foreground_ratio)

        texture_resolution = int(job_input.get("texture_resolution", 1024))
        remesh_option = job_input.get("remesh_option", "none")
        if remesh_option not in ("none", "triangle", "quad"):
            raise ValueError("remesh_option должен быть 'none', 'triangle' или 'quad'")
        target_vertex_count = int(job_input.get("target_vertex_count", -1))

        with torch.no_grad():
            autocast_ctx = (
                torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
                if "cuda" in DEVICE
                else nullcontext()
            )
            with autocast_ctx:
                mesh, _ = model.run_image(
                    [image],
                    bake_resolution=texture_resolution,
                    remesh=remesh_option,
                    vertex_count=target_vertex_count,
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "mesh.glb")
            mesh.export(out_path, include_normals=True)
            result = _upload_or_encode(out_path)

        result["format"] = "glb"
        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
