# Module template

Модуль предназначен для определения тональности текста (негативная, нейтральная или позитивная).

## Структура

    .
    ├── data                                # Данные (примеры, необходимые маленькие файлы, инструкции по датасетам)
    │   │
    │   ├── input_example.json              # Пример входных данных
    │   └── output_example.json             # Пример выходных данных
    │
    ├── models                              # Реализации конкретных моделей
    │   │ 
    │   └── XLMRobertaBaseSentiment         # Обертка для модели "xlm_roberta_base-ru-sentiment-sentirueval2016"
    │       │
    │       ├── model                       # Детали реализации, служебные классы
    │       ├── checkpoints                 # Директория для подгрузки сохраненной модели по умолчанию
    │       ├── data                        # Данные для обучения и валидации моделей
    │       ├── scripts                     # Скрипты для подготовки данных, обучения и валидации
    │       └── README.md                   # Подробное описание модели
    │ 
    ├── service.py                          # Flask-app, предоставляющий сервис для вызова моделей по API
    ├── Dockerfile                          # Dockerfile для запуска модуля как сервиса
    ├── requirements.txt                    # Локальные зависимости модуля
    └── README.md (вы здесь)                # Описание модуля

## Сборка и запуск сервиса

Предобученные модели можно скачать по ссылке https://disk.yandex.ru/d/i2vzCR1N-kxOzg и переместить в соответствующие папки:

* XLMRobertaBaseSentirueval2016/sismetanin/xlm_roberta_base-ru-sentiment-sentirueval2016 --> ./models/XLMRobertaBaseSentirueval2016/checkpoints/sismetanin/xlm_roberta_base-ru-sentiment-sentirueval2016

Если во время создания Docker-образа не будут обнаружены файлы с моделями, они будут загружены автоматически.

Также может быть применена оптимизация с установкой PyTorch - можно заранее скачать его по ссылке https://disk.yandex.ru/d/j8FuKiGS0S0vRw и разместить в директории ./models

Собираем докер-образ с сервисом:
```commandline
docker build -t sentiment_analysis:0.1 .
```
Запускаем контейнер фоновым процессом:
```commandline
docker run -d -p 8805:8805 --name sentiment_analysis sentiment_analysis:0.1
```

## Вызов сервиса

Для использования модуля как отдельного сервиса нужно отправлять
json файл на endpoint "/recognize" с помощью POST-запроса. Например, 
это можно сделать с помощью библиотеки requests и скрипта на python:

```python
import requests
import json

filename = "./data/input_example.json"
input_json = json.loads(open(filename, encoding="utf-8").read())
resp = requests.post('http://localhost:8805/recognize', 
                     data=json.dumps(input_json), 
                     headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
```

#### Входные данные: объект json

Входной JSON должен обязательно содержать элемент "summaries", который содержит список строк:

```
{
    "diarization": [
      ...   
    ],
    "text": "Первый сегмент. Второй сегмент.",
    "segments": ["Первый сегмент.", "Второй сегмент."],
    "summaries": ["Сегмент", "Сегмент"],
}
```

Пример входных данных:

```data/input_example.json```

#### Выходные данные: объект json

Добавляет элемент "sentiments" в исходный JSON

```
{
    "diarization": [
      ...   
    ],
    "text": "Первый сегмент. Второй сегмент.",
    "segments": ["Первый сегмент.", "Второй сегмент."],
    "summaries": ["Сегмент", "Сегмент"],
    "summaries": ["Сегмент", "Сегмент"],
    "sentiments": ["NEUTRAL", "NEUTRAL"],
}
```

Пример выходных данных:

```data/output_example.json```

## Модели

#### Поддерживаемые модели

* **RuBERTSentiment** - пока что основная модель для анализа тональности русских новостных текстов

#### Обучение моделей

Ссылки на скрипты для подготовки данных, обучения и тестирования моделей расположены в директориях scripts соответствующих моделей

#### Результаты тестирования

Тестирование было проведено разработчиком модели и не опубликовано. По косвенным признаком можно судить об удовлетворительном качестве модели
