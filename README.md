# Выбор лучшей подписи к изображению

Проект по разработке текстового и визуального модулей для выбора наилучшей подписи из нескольких сгенерированных кандидатов, в рамках проекта по Основам глубокого обучения 2 курса ПАДИИ ВШЭ СПб.

Выполнили: Казарин Евгений, Тищенко Софья

## Структура

- `notebooks/`: ноутбуки с подготовкой данных, обучением и объединением модулей (код + графики)
- `models/`: веса обученных моделей (кроме cross-encoder, его веса тяжелые)
- `data/`: исходный датасет и производные таблицы метрик

## Данные

Источник: `nikhil7280/coco-image-caption` (Kaggle), на основе MS COCO train2014.
Все скрипты ожидают файлы в `data/`:
- `dataset_df.csv` - пути к изображениям, 5 BLIP-подписей и 5 COCO-подписей на каждое
- `captions_metrics.csv` - CIDEr/BLEU/METEOR для каждой BLIP-подписи относительно референсов
Также во время исполнения сохраняются дополнительные файлы csv, которые мы не включили в репозиторий для экономии памяти.
## notebooks/

| Ноутбук                | Содержание                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| `BLIPInitialize.ipynb` | генерация BLIP-кандидатов, подсчёт CIDEr/BLEU/METEOR                                              |
| `Text_Module.ipynb`    | обучение и сравнение текстовых ранкеров (bi-encoder, cross-encoder, baseline, LLM, bi + features) |
| `xLstmTrain.ipynb`     | обучение xLSTM-ранкера с нуля, отдельно от остальных текстовых моделей                            |
| `Visual_Module.ipynb`  | визуальный модуль: CLIP score, grounding-признаки, RankNet                                        |
| `DeepLearning_U.ipynb` | объединение текстового и визуального модулей, финальный пайплайн и метрики                        |

Ноутбуки открываются через `jupyter notebook` / `jupyter lab` / Colab. 

## models/

```
models/
├── bi_encoder_head.pt.gz       # MiniLM (заморожен) + MLP-голова
├── cross_encoder_scorer.pt.gz  # (НЕ ВКЛЮЧЕН В РЕПОЗИТОРИЙ, 407мб) RoBERTa-base, дообучена целиком + MLP-голова
├── xlstm_scorer.pt.gz          # xLSTM, обучена с нуля + MLP-голова
└── visual_ranknet_model.pt.gz  # RankNet на CLIP score + grounding-признаках
```

Файлы сжаты (`.pt.gz`), перед загрузкой распаковать (`gzip -d`).

## Запуск

Из `notebooks/`, по порядку:

```bash
jupyter notebook BLIPInitialize.ipynb
jupyter notebook Text_Module.ipynb
jupyter notebook xLstmTrain.ipynb
jupyter notebook Visual_Module.ipynb
jupyter notebook DeepLearning_U.ipynb
```

Зависимости: `pandas`, `numpy`, `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `nltk`, `pycocoevalcap`.

## Результаты

Текстовый модуль (CIDEr выбранной подписи, полный тест):

| Метод         | CIDEr | Top-1 Acc |
| ------------- | ----- | --------- |
| Oracle        | 1.054 | 1.000     |
| Cross-encoder | 0.871 | 0.351     |
| Bi-encoder    | 0.861 | 0.331     |
| xLSTM         | 0.859 | 0.338     |
| Random        | 0.761 | 0.201     |

Объединение модулей снижает CIDEr с 0.871 до 0.840, но почти вдвое уменьшает долю подписей с неподтверждёнными изображением деталями: с 38.6% до 18.4%.