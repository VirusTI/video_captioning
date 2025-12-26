# Генерация описания к видео

**Ратков Виктор Андреевич** (4 курс, Б05-227)

## Постановка задачи

Дано короткое видео, нужно сгенерировать осмысленное текстовое описание происходящего на английском языке. Цель проекта — демонстрация того, что маленькая по числу обучаемых параметров модель при правильной архитектуре и хороших данных способна давать достаточно робастные и полезные результаты.

На практике такую модель можно применять для генерации человеко-читаемого признакового описания больших объёмов видеоданных.

## Формат входных и выходных данных

- **Вход:** видеоролик длиной 10–30 секунд в формате mp4
- **Выход:** описание — строка на английском языке

## Метрики

Для оценки качества генерации описаний планируется использовать стандартные метрики для видео/имидж captioning:

- **BLEU-4** — качество n-gram совпадений;
- **METEOR** — усовершенствованный F1-score для слов в описании с точностью до синонимов;

## Валидация и тест

Будем использовать разбиение 80% train, 10% validation и 10% test. Чтобы сделать результат воспроизводимым, будем фиксировать `random_seed` для разбиения и обучения. Также сохраним индексы семплов из каждой выборки.

## Датасеты

Основной датасет — MSR-VTT (Microsoft Research Video to Text, https://huggingface.co/datasets/friedrichor/MSR-VTT):

- 10 000 видеоклипов общей длительностью около 41 часа;
- 20 текстовых аннотаций на английском для каждого клипа (200 000 пар «клип-подпись»);
- 20 тематических категорий (спорт, музыка, новости, анимация и др.);

### Нюансы

- Все видео имеют довольно низкое качество и низкий FPS, что позволяет уместить весь датасет в ~2 ГБ, но в то же время может сказаться на качестве распознавания быстро движущихся или мелких объектов.
- Подписи были получены от более тысячи разметчиков, и никак не проверялись, поэтому потенциально могут иметь шум.
- Большое разнообразие сцен и объектов может усложнить сходимость небольших моделей, так как потребует обобщения широкого домена данных.

## Внедрение

Репозиторий будет оформлен как Python-пакет с использованием UV как менеджера зависимостей. Будет реализован CLI-интерфейс для обучения и тестирования, а также Docker-сервис с API для удобного внедрения модели.

---

# Video Captioning (реализация)

Этот репозиторий содержит реализацию двух моделей (бейзлайн и основная модель) для генерации описаний видео на английском языке, а также пайплайн подготовки данных, обучения, инференса и подготовки к продакшену (ONNX / TensorRT / Triton).

## Результаты

Ниже приведены результаты на валидации (для ориентира):

| Модель | BLEU-4 | METEOR |
| --- | ---: | ---: |
| Бейзлайн (ResNet-50 + LSTM) | 11.3% | 13.2% |
| Основная модель (ViT-tiny + Transformer) | 15.2% | 18.1% |

## Быстрый старт

### 1) Установка

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd video-captioning

# Устанавливаем зависимости (включая dev-инструменты для pre-commit)
uv sync --python 3.11 --dev
```

### 2) Проверка качества кода

```bash
uv run pre-commit install
uv run pre-commit run -a
```

### 3) Подготовка данных

Данные скачиваются из открытого источника (HuggingFace). DVC используется как воспроизводимый пайплайн и для фиксации контрольных сумм выходных артефактов (см. `dvc.lock`).

```bash
HF_HUB_DISABLE_XET=1 uv run dvc repro setup_data
```

Что делает `setup_data.py`:
- скачивает `MSRVTT_Videos.zip` и распаковывает видео в `data/videos/video/*.mp4`
- скачивает метаданные train/test и готовит разбиение train/val/test
- сохраняет сплиты (`data/*_split.json`) и `data/dataset_summary.json`
- выполняет best-effort проверку целостности zip (сравнение SHA256 с HuggingFace etag, если доступно)

```

### 4) Обучение

```bash
# Бейзлайн
python -m video_captioning.commands train_baseline

# Основная модель
python -m video_captioning.commands train_advanced
```
Рекомндуется для проверки запускать обучение бейзлайна, так как оно в 5 раз быстрее.

Примеры Hydra overrides:

```bash
python -m video_captioning.commands train_baseline training.max_epochs=3 training.batch_size=16
python -m video_captioning.commands train_advanced training.max_epochs=3 training.batch_size=8
```

### 5) Инференс

Формат входа: mp4 (10–30 секунд). Выход: строка на английском.

```bash
python -m video_captioning.commands inference baseline /path/to/model.ckpt /path/to/video.mp4
python -m video_captioning.commands inference advanced /path/to/model.ckpt /path/to/video.mp4
```

## Структура репозитория (подробно)

Ниже — ориентир по ключевым директориям/файлам и тому, за что они отвечают:

```text
video-captioning/
    README.md                    # документация проекта (как запустить, что где лежит)
    pyproject.toml               # зависимости/скрипты проекта (uv)
    uv.lock                      # lock-файл окружения (детерминированные версии)

    setup_data.py                # скачивание MSR-VTT + подготовка сплитов
    dvc.yaml                     # DVC-пайплайн (стейдж setup_data)
    dvc.lock                     # зафиксированные хэши выходов стейджа (воспроизводимость)
    .dvc/                        # служебные файлы DVC (локальные)
    .dvcignore                   # что DVC игнорирует (например, большие видео)

    configs/                     # конфигурации Hydra
        baseline.yaml              # конфиг обучения бейзлайна
        advanced.yaml              # конфиг обучения основной модели
        dataset/msr_vtt.yaml       # параметры датасета/путей/сплитов
        model/baseline.yaml        # архитектура бейзлайна
        model/advanced.yaml        # архитектура основной модели
        training/trainer.yaml      # параметры Lightning trainer

    video_captioning/            # основной python-пакет проекта
        commands.py              # CLI-команды (train/infer/export)

        data/                    # подготовка данных и dataloading
            dataset.py           # Dataset/DataModule, загрузка видео/токенизация

        models/                  # архитектуры и forward/генерация
            model.py             # baseline/advanced модели

        training/                # обучение (Lightning)
            train_baseline.py    # тренировка бейзлайна (entrypoint)
            train_advanced.py    # тренировка основной модели (entrypoint)
            lightning_module.py  # LightningModule для baseline
            lightning_module_advanced.py  # LightningModule для advanced
            val_logging.py       # логирование примеров/валид. генерации

        inference/               # инференс и постобработка
            inference.py         # загрузка чекпоинта + генерация caption

        onnx/                    # экспорт в ONNX
            export.py            # экспорт baseline/advanced в ONNX

        evaluation/              # метрики качества
            metrics.py           # BLEU-4, METEOR и др.

        utils/                   # общие утилиты
            config.py            # работа с Hydra/OmegaConf
            logging.py           # логирование
            experiment_tracking.py  # трекинг экспериментов
            mlflow_export.py     # экспорт артефактов/графиков

    scripts/                     # инфраструктурные скрипты
        build_trt_engine.sh        # сборка TensorRT engine из ONNX (через контейнер)
        run_triton_server.sh       # запуск Triton server (ONNX/TensorRT)
        triton_smoke.py            # smoke-клиент для Triton (HTTP)
        check_*.py                 # проверки окружения/данных/конфигов

    artifacts/                   # основные артефакты (чекпоинты/экспорт/репозиторий для Triton)
        checkpoints/               # .ckpt файлы Lightning
        baseline_final.pt          # финальный экспорт (пример)
        advanced_final.pt          # финальный экспорт (пример)
        tokenizers/                # BPE-токенизатор (SentencePiece)
        onnx/                      # ONNX-модели
        trt/                       # TensorRT engines
        triton_model_repo/         # готовая структура model_repository для Triton

    data/                        # локальные данные/сплиты/видео (частично генерируется)
        *_split.json               # индексы train/val/test
        dataset_summary.json
        videos/                    # распакованные видео (обычно не трекаются DVC)

    models/                      # дополнительные файлы моделей (например, BPE) — если используются
    plots/                       # экспорт графиков/отчетов
```

Примечания:
- Большие данные (видео) живут в `data/videos/` и скачиваются скриптом `setup_data.py`.
- Для воспроизводимости сплиты и summary фиксируются через DVC (`dvc.yaml` + `dvc.lock`).
- Чекпоинты обучения по умолчанию складываются в `artifacts/checkpoints/` (см. `logging.output_dir` в конфигах).

## Архитектуры

### Бейзлайн: ResNet-50 + LSTM

Идея: использовать 1 кадр как представление видео.

- Визуальный энкодер: предобученный ResNet-50, заморожен.
- Из видео извлекается один кадр, преобразуется к `224x224`.
- Эмбеддинг кадра подаётся в LSTM-декодер, который по teacher-forcing учится предсказывать следующий токен.
- На инференсе используется генерация (greedy / beam / top-k — выбирается конфигом).

### Основная модель: ViT-tiny + Transformer

Идея: учитывать временную структуру видео (несколько кадров).

- Визуальный энкодер: ViT-tiny/16 (заморожен).
- Из видео декодируется последовательность кадров, далее идёт семплинг по FPS (например, 2–4 fps).
- Получается тензор кадров `T x 3 x 224 x 224` + маска кадров (`frame_mask`) для паддинга.
- Темпоральный блок и/или Transformer-слои агрегируют последовательность эмбеддингов кадров.
- Текстовый декодер (Transformer) генерирует подпись.

## Подготовка к продакшену

### ONNX

```bash
uv sync --python 3.11 --extra export

python -m video_captioning.commands convert_onnx baseline \
    /path/to/baseline.ckpt artifacts/onnx/baseline.onnx

python -m video_captioning.commands convert_onnx advanced \
    /path/to/advanced.ckpt artifacts/onnx/advanced.onnx
```

### TensorRT

Требования:
- NVIDIA GPU + драйвер
- Docker + NVIDIA Container Toolkit (`docker run --gpus=all ...`)

```bash
# baseline
bash scripts/build_trt_engine.sh artifacts/onnx/baseline.onnx

# advanced
sudo bash scripts/build_trt_engine.sh artifacts/onnx/advanced.onnx --fp32
```

Примечание: в текущем окружении advanced TensorRT в FP16 может давать NaN, поэтому используется FP32.

### Triton Inference Server

```bash
# ONNX
bash scripts/run_triton_server.sh baseline onnx artifacts/onnx/baseline.onnx
bash scripts/run_triton_server.sh advanced onnx artifacts/onnx/advanced.onnx

# TensorRT
bash scripts/run_triton_server.sh advanced trt artifacts/trt/advanced/advanced.plan
```

Проверка smoke-тестом:

```bash
uv run python scripts/triton_smoke.py main baseline --url=http://localhost:8000
uv run python scripts/triton_smoke.py main advanced --url=http://localhost:8000
```

## Конфигурация (Hydra)

Основные конфиги лежат в `configs/` и сгруппированы по компонентам:

```text
configs/
    dataset/msr_vtt.yaml
    model/{baseline,advanced}.yaml
    training/trainer.yaml
    {baseline,advanced}.yaml
```

## Логирование

- Логируются лоссы и метрики (BLEU-4, METEOR и др.)
- Поддерживается MLflow (по умолчанию `http://127.0.0.1:8080` задаётся в конфиге)
- Графики экспортируются в директорию `plots/`
