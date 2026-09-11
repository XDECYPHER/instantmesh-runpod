"""
RunPod Serverless handler для TripoSR (image -> 3D mesh).

Ожидаемый вход (event["input"]):
{
    "image_base64": "<base64 картинки>",   # либо
    "image_url": "https://...",            # одно из двух обязательно
    "remove_background": true,             # опционально, по умолчанию true
    "foreground_ratio": 0.85,              # опционально
    "mc_resolution": 256,                  # опционально, разрешение marching cubes
    "output_format": "glb"                 # glb | obj
}

Выход:
{
    "output": {
        "mesh_base64": "...",       # если S3 не настроен
        "mesh_url": "...",          # если настроен bucket через env RunPod
        "format": "glb"
    }
}
"""

import os
import sys
import io
import base64
import traceback
import tempfile

import runpod
import torch
import numpy as np
import requests
from PIL import Image

# --- Делаем доступным код самого TripoSR ------------------------------------
sys.path.append("/workspace/TripoSR")

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/workspace/model_cache"

# =============================================================================
# Модель грузим ОДИН РАЗ при старте воркера (не внутри handler!),
# чтобы cold start не платил за загрузку весов на каждый запрос повторно
# в рамках одного и того же "тёплого" контейнера.
# =============================================================================
print(f"[TripoSR] Загружаю модель на устройство: {DEVICE}")
model = TSR.from_pretrained(
    MODEL_DIR,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(DEVICE)
print("[TripoSR] Модель загружена и готова к работе.")


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


def _prepare_image(image: Image.Image, job_input: dict) -> Image.Image:
    """Удаление фона + нормализация под ожидания TripoSR."""
    do_remove_bg = job_input.get("remove_background", True)
    foreground_ratio = float(job_input.get("foreground_ratio", 0.85))

    if do_remove_bg:
        image = remove_background(image)
        image = resize_foreground(image, foreground_ratio)
        image_arr = np.array(image).astype(np.float32) / 255.0
        image_arr = image_arr[:, :, :3] * image_arr[:, :, 3:4] + (1 - image_arr[:, :, 3:4]) * 0.5
        image = Image.fromarray((image_arr * 255.0).astype(np.uint8))
    else:
        image = image.convert("RGB")

    return image


def _upload_or_encode(file_path: str) -> dict:
    """
    Если у воркера настроен bucket (стандартные RunPod env-переменные для
    сетевого volume/S3), заливаем файл и возвращаем ссылку.
    Иначе — отдаём файл как base64 прямо в ответе (ок для небольших мешей).
    """
    try:
        from runpod.serverless.utils import rp_upload

        bucket_configured = os.environ.get("BUCKET_ENDPOINT_URL")
        if bucket_configured:
            url = rp_upload.upload_file_to_bucket(
                file_name=os.path.basename(file_path),
                file_location=file_path,
            )
            return {"mesh_url": url}
    except Exception as e:
        print(f"[TripoSR] Не удалось загрузить в bucket, fallback на base64: {e}")

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return {"mesh_base64": encoded}


def handler(event):
    try:
        job_input = event.get("input", {})

        image = _load_image(job_input)
        image = _prepare_image(image, job_input)

        mc_resolution = int(job_input.get("mc_resolution", 256))
        output_format = job_input.get("output_format", "glb").lower()
        if output_format not in ("glb", "obj"):
            raise ValueError("output_format должен быть 'glb' или 'obj'")

        with torch.no_grad():
            scene_codes = model([image], device=DEVICE)
            meshes = model.extract_mesh(scene_codes, resolution=mc_resolution, has_vertex_color=False)

        mesh = meshes[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, f"mesh.{output_format}")
            mesh.export(out_path)
            result = _upload_or_encode(out_path)

        result["format"] = output_format
        return {"output": result}

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
