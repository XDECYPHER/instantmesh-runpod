# TripoSR на RunPod Serverless

Картинка → 3D-модель (.glb / .obj) через [TripoSR](https://github.com/VAST-AI-Research/TripoSR),
задеплоенный как serverless endpoint на RunPod.

## Структура репозитория

```
.
├── Dockerfile                      # образ с CUDA, torch, TripoSR и весами модели
├── handler.py                      # точка входа для RunPod (обработка запроса)
├── requirements.txt                # для локального теста вне Docker
├── test_input.json                 # пример входных данных
├── .github/workflows/docker-build.yml  # автосборка образа в GHCR
├── .dockerignore
└── .gitignore
```

## Шаг 1. Выложить репозиторий на GitHub

```bash
cd triposr-runpod
git init
git add .
git commit -m "TripoSR RunPod serverless worker"
git branch -M main
git remote add origin https://github.com/<твой_юзернейм>/<репо>.git
git push -u origin main
```

После пуша GitHub Actions (`.github/workflows/docker-build.yml`) автоматически
соберёт Docker-образ и запушит его в **GitHub Container Registry**:
`ghcr.io/<твой_юзернейм>/<репо>:latest`

Прогресс сборки смотри во вкладке **Actions** репозитория.
Сборка первый раз идёт долго (10-20 минут) — образ тяжёлый (CUDA + torch + веса модели).

⚠️ **Важно:** после первой успешной сборки зайди в GitHub →
`Packages` → выбери свой пакет → **Package settings** → сделай его **Public**
(иначе RunPod не сможет забрать образ без доп. настройки авторизации в registry).

## Шаг 2. Создать Serverless Endpoint на RunPod

1. Зайди на [runpod.io](https://www.runpod.io/) → **Serverless** → **New Endpoint**
2. **Container Image**: `ghcr.io/<твой_юзернейм>/<репо>:latest`
3. **GPU**: бери минимум 16GB VRAM (например RTX A4000 / A5000 / 4090) —
   TripoSR не гигантская модель, но с запасом надёжнее
4. **Container Disk**: не меньше 25-30 GB (образ тяжёлый из-за CUDA+torch+весов)
5. **Max Workers**: сколько хочешь параллельных воркеров (для старта хватит 1-2)
6. Остальное можно оставить по умолчанию → **Deploy**

RunPod сам подтянет образ и запустит воркер по первому запросу (cold start).

## Шаг 3. Проверка / вызов эндпоинта

После деплоя у тебя будет **Endpoint ID** и **API Key** в личном кабинете RunPod.

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "image_url": "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png",
          "output_format": "glb"
        }
      }'
```

В ответ придёт JSON с полем `output.mesh_base64` (готовый .glb файл в base64)
либо `output.mesh_url`, если ты настроил bucket для аплоада (см. ниже).

Чтобы сохранить файл из base64 локально:

```bash
python3 -c "
import base64, json
data = json.load(open('response.json'))
with open('result.glb', 'wb') as f:
    f.write(base64.b64decode(data['output']['mesh_base64']))
"
```

## Опционально: заливка результата в S3-совместимый bucket

Если меши крупные и гонять их base64 в JSON неудобно — настрой в RunPod
**Network Volume / S3 credentials** и добавь env-переменную `BUCKET_ENDPOINT_URL`
в настройках эндпоинта. Handler сам определит её наличие и загрузит файл туда
вместо base64 (см. функцию `_upload_or_encode` в `handler.py`).

## Параметры запроса (`input`)

| Поле                | Обязательно | По умолчанию | Описание                                  |
|---------------------|-------------|--------------|--------------------------------------------|
| `image_base64`      | одно из двух с `image_url` | —   | Картинка в base64                          |
| `image_url`         | одно из двух с `image_base64` | — | Ссылка на картинку                         |
| `remove_background` | нет         | `true`       | Удалять ли фон (rembg) перед реконструкцией |
| `foreground_ratio`  | нет         | `0.85`       | Насколько объект заполняет кадр после кропа |
| `mc_resolution`     | нет         | `256`        | Разрешение marching cubes (больше = детальнее и медленнее) |
| `output_format`     | нет         | `glb`        | `glb` или `obj`                            |

## Частые проблемы (и почему "100% с первого раза" не бывает)

- **Сборка `torchmcubes` падает** — нужен образ `-devel`, а не `-runtime`
  (в Dockerfile это уже учтено), т.к. пакет компилируется с CUDA-кодом.
- **Out of memory на GPU** — уменьши `mc_resolution` (например до 128) или
  возьми GPU с большим VRAM.
- **Долгий cold start** — образ тяжёлый (~15-20GB с CUDA+torch+весами).
  Это нормально для первого запроса после простоя; помогает держать
  `Min Workers: 1`, если готов платить за "разогретый" воркер.
- **Пакет в GHCR приватный** — RunPod не может его скачать. Сделай Public
  (см. Шаг 1) или пропиши registry credentials в настройках эндпоинта.

## Локальный тест (если есть GPU)

```bash
docker build -t triposr-worker .
docker run --gpus all -it --rm triposr-worker python3 handler.py --test_input test_input.json
```
