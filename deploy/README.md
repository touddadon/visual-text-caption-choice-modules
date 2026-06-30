# Deploy

Каталог `deploy/` содержит отдельное Gradio-приложение для запуска проекта без обучающих ноутбуков.

## Что делает приложение

1. Принимает изображение.
2. Генерирует несколько вариантов подписи через BLIP.
3. Оценивает подписи одним из методов:
   - `cross_encoder` — основной текстовый скорер;
   - `bi_encoder` — более лёгкий текстовый скорер;
   - `consensus` — выбор подписи, наиболее похожей на остальные кандидаты;
   - `first_blip` — базовый вариант, первая BLIP-подпись.
4. Возвращает лучшую подпись и таблицу всех кандидатов.

## Структура

```text
deploy/
├── app.py
├── modeling.py
├── requirements.txt
├── Dockerfile
├── prepare_models.py
├── .gitattributes
├── .gitignore
└── models/
    └── README.md
```

## Локальный запуск

Из корня репозитория:

```bash
cd deploy
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Открыть в браузере:

```text
http://127.0.0.1:7860
```

## Как положить веса

Если файлы весов лежат в корне проекта рядом с `deploy/`, выполнить:

```bash
cd deploy
python prepare_models.py
```

После этого должны появиться:

```text
deploy/models/cross_encoder_scorer.pt
deploy/models/bi_encoder_head.pt
```

## Загрузка на Git

Из корня репозитория:

```bash
git lfs install
git add deploy/
git commit -m "Add deployment app"
git push
```

`cross_encoder_scorer.pt` большой, поэтому для него нужен Git LFS. Без Git LFS GitHub обычно не примет файл больше 100 MB.

## Docker-запуск

```bash
cd deploy
docker build -t image-caption-enhancer .
docker run --rm -p 7860:7860 image-caption-enhancer
```
