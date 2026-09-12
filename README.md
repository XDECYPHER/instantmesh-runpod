# Stable Fast 3D (SF3D) на RunPod Serverless

Картинка → 3D-модель (.glb с текстурой) через [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d).

## Перед сборкой

1. Зайди на https://huggingface.co/stabilityai/stable-fast-3d и нажми "Agree and access repository"
2. Создай токен с правами Read: https://huggingface.co/settings/tokens
3. Добавь его в GitHub: Settings → Secrets and variables → Actions → `HF_TOKEN`

## Структура

- `Dockerfile` — образ с CUDA, torch, SF3D и весами модели
- `handler.py` — точка входа для RunPod
- `.github/workflows/docker-build.yml` — автосборка в GHCR

## Параметры запроса (input)

| Поле | По умолчанию | Описание |
|---|---|---|
| `image_base64` / `image_url` | — | картинка (одно из двух) |
| `foreground_ratio` | 0.85 | заполнение кадра объектом |
| `texture_resolution` | 1024 | разрешение текстуры |
| `remesh_option` | "none" | none / triangle / quad |
| `target_vertex_count` | -1 | без ограничения полигонов |

Лицензия модели: Stability AI Community License — бесплатно для дохода < $1M/год.
