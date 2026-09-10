import base64
import io
import os
import tempfile

import numpy as np
import rembg
import runpod
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Загружаем модель TripoSR, это займёт какое-то время при первом запуске...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()
print(f"Модель загружена на {device}, воркер готов принимать задачи")


def fill_background(image: Image.Image) -> Image.Image:
    """Так же, как в официальном демо: заполняет прозрачный фон серым."""
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def handler(job):
    job_input = job["input"]
    image_b64 = job_input["image_base64"]
    mc_resolution = job_input.get("mc_resolution", 256)
    remove_bg = job_input.get("remove_background", True)

    image_bytes = base64.b64decode(image_b64)
    input_image = Image.open(io.BytesIO(image_bytes))

    if remove_bg:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, 0.9)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)

    print("Запускаем реконструкцию 3D...")
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        mesh.export(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        model_bytes = f.read()
    os.remove(tmp_path)

    print("Готово, отправляем результат")
    return {
        "model_base64": base64.b64encode(model_bytes).decode("utf-8"),
        "format": "glb",
    }


runpod.serverless.start({"handler": handler})
