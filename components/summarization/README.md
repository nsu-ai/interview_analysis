# Module template

Модуль предназначен для генерации краткой информативной версии текста.

## Структура

    .
    ├── data                        # Данные (примеры, необходимые маленькие файлы, инструкции по датасетам)
    │   │
    │   ├── input_example.json      # Пример входных данных
    │   └── output_example.json     # Пример выходных данных
    │
    ├── models                      # Реализации конкретных моделей
    │   │ 
    │   └── MBartRuSumGazeta        # Обертка для модели "IlyaGusev/mbart_ru_sum_gazeta"
    │       │
    │       ├── model               # Детали реализации, служебные классы
    │       ├── checkpoints         # Директория для подгрузки сохраненной модели по умолчанию
    │       ├── data                # Данные для обучения и валидации моделей
    │       ├── scripts             # Скрипты для подготовки данных, обучения и валидации
    │       └── README.md           # Подробное описание модели
    │ 
    ├── service.py                  # Flask-app, предоставляющий сервис для вызова моделей по API
    ├── Dockerfile                  # Dockerfile для запуска модуля как сервиса
    ├── requirements.txt            # Локальные зависимости модуля
    └── README.md (вы здесь)        # Описание модуля

## Сборка и запуск сервиса

Предобученные модели можно скачать по ссылке https://disk.yandex.ru/d/ELYZECp-FHMyAQ и переместить в соответствующие папки:

* MBartRuSumGazeta/IlyaGusev_mbart_ru_sum_gazeta --> ./models/MBartRuSumGazeta/checkpoints/IlyaGusev_mbart_ru_sum_gazeta

Если во время создания Docker-образа не будут обнаружены файлы с моделями, они будут загружены автоматически.

Также может быть применена оптимизация с установкой PyTorch - можно заранее скачать его по ссылке https://disk.yandex.ru/d/j8FuKiGS0S0vRw и разместить в директории ./models

Собираем докер-образ с сервисом:
```commandline
docker build -t summarization:0.1 .
```
Запускаем контейнер фоновым процессом:
```commandline
docker run -d -p 8807:8807 --name summarization summarization:0.1
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
resp = requests.post('http://localhost:8807/recognize', 
                     data=json.dumps(input_json), 
                     headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
```

#### Входные данные: объект json

Входной JSON должен обязательно включать элемент "segments", который содержит список строк:

```
{
    "diarization": [
      ...   
    ],
    "text": "Первый сегмент. Второй сегмент.",
    "segments": ["Первый сегмент.", "Второй сегмент."]
}
```

Пример входных данных:

```data/input_example.json```

#### Выходные данные: объект json

Добавляет элемент "summaries" в исходный JSON, содержащее краткие информативные версии сегментов:

```
{
    "diarization": [
      ...   
    ],
    "text": "Первый сегмент. Второй сегмент.",
    "segments": ["Первый сегмент.", "Второй сегмент."],
    "summarizes": ["Сегмент", "Сегмент"],
}
```

Пример выходных данных:

```data/output_example.json```

## Модели

#### Поддерживаемые модели

* **MbartRuSumGazeta** - пока что основная модель для суммаризации русских новостных текстов

#### Обучение моделей

Ссылки на скрипты для подготовки данных, обучения и тестирования моделей расположены в директориях scripts соответствующих моделей
#### Результаты тестирования

Тестирование было проведено разработчиком модели.

| Метрика | MbartRuSumGazeta |
|--------|:----------------:|
| R-1-f1 |       32.4       |
| R-2-f1 |       14.3       |
| R-L-f1 |       28.0       |
| chrF |       39.7       |
| METEOR |       26.4       |
| BLEU |       12.1       |
| Avg char length |       371        |
