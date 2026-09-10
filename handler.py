import os
import glob
import base64
import uuid
import subprocess

import requests
import runpod
import trimesh

INSTANTMESH_DIR = "/workspace/InstantMesh"


def download_image(url: str, save_path: str) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)


def handler(event):
    values = event.get("input", {})
    image_url = values.get("image_url")
    if not image_url:
        return {"error": "Поле 'image_url' обязательно в input"}

    seed = values.get("seed", 42)
    export_texmap = values.get("export_texmap", False)
    no_rembg = values.get("no_rembg", False)

    job_id = uuid.uuid4().hex
    input_path = f"/tmp/{job_id}.png"
    output_dir = f"/tmp/out_{job_id}/"

    try:
        download_image(image_url, input_path)
    except Exception as e:
        return {"error": f"Не удалось скачать картинку по image_url: {e}"}

    cmd = [
        "python", "run.py",
        "configs/instant-mesh-large.yaml",
        input_path,
        "--output_path", output_dir,
        "--seed", str(seed),
    ]
    if export_texmap:
        cmd.append("--export_texmap")
    if no_rembg:
        cmd.append("--no_rembg")

    result = subprocess.run(
        cmd,
        cwd=INSTANTMESH_DIR,
        capture_output=True,
        text=True,
        timeout=280,
    )

    if result.returncode != 0:
        return {
            "error": "InstantMesh (run.py) завершился с ошибкой",
            "stderr": result.stderr[-3000:],
        }

    # Официальная структура вывода run.py:
    # {output_path}/instant-mesh-large/meshes/{имя_картинки_без_расширения}.obj
    obj_pattern = os.path.join(output_dir, "instant-mesh-large", "meshes", "*.obj")
    obj_files = glob.glob(obj_pattern)
    if not obj_files:
        return {
            "error": "Файл .obj не найден после генерации — что-то пошло не так внутри run.py",
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-2000:],
        }

    obj_path = obj_files[0]
    glb_path = obj_path.replace(".obj", ".glb")

    try:
        mesh = trimesh.load(obj_path, force="mesh")
        mesh.export(glb_path)
    except Exception as e:
        return {"error": f"Не удалось сконвертировать .obj в .glb: {e}"}

    with open(glb_path, "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"model_base64": model_b64, "format": "glb"}


runpod.serverless.start({"handler": handler})
