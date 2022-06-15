# Module template

Модуль предназначен для разделения цельного текста на блоки - группы предложений, объединенные общим смыслом


## Структура

    .
    ├── data                        # Данные (примеры, необходимые маленькие файлы, инструкции по датасетам)
    │   │
    │   ├── input_example.json      # Пример входных данных
    │   └── output_example.json     # Пример выходных данных
    │
    ├── models                      # Реализации конкретных моделей
    │   │ 
    │   └── HierBERTSegmentator     # Наша основная модель для семантической сегментации текстов
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

Предобученные модели можно скачать по ссылке https://disk.yandex.ru/d/p7FcSEmJFHh9zA и переместить в соответствующие папки:

* **HierBERTSegmentator/checkpoint-4000** --> **./models/HierBERTSegmentator/checkpoints/checkpoint-4000**
* **HierBERTSegmentator/sentence_ru_cased_L-12_H-768_A-12_pt** --> **models/HierBERTSegmentator/checkpoints/sentence_ru_cased_L-12_H-768_A-12_pt**

Если во время создания Docker-образа не будут обнаружены файлы с моделями, они будут загружены автоматически.

Также может быть применена оптимизация с установкой PyTorch - можно заранее скачать его по ссылке https://disk.yandex.ru/d/j8FuKiGS0S0vRw 
и разместить в директории **./models**

Собираем докер-образ с сервисом:
```commandline
docker build -t segmentation:0.1 .
```
Запускаем Контейнер фоновым процессом:
```commandline
docker run -d -p 8806:8806 --name segmentation segmentation:0.1
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
resp = requests.post('http://localhost:<port>/recognize', 
                     data=json.dumps(input_json), 
                     headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
```

#### Входные данные: объект json

Входной JSON должен обязательно содержать элемент "text", который содержит некоторый текст со знаками препинания 
(как минимум, точками - для разбиения на предложения):

```
{
    "diarization": [
      ...   
    ],
    "text": "Текст с пунктуацией."
}
```

Пример входных данных:

```data/input_example.json```

#### Выходные данные: объект json

Добавляет список "segments" в исходный JSON


```
{
    "diarization": [
      ...   
    ],
    "text": "Текст с пунктуацией.",
    "segments": ["text1", "text2", "text3", ...]
}
```

Пример выходных данных:

```data/output_example.json```

## Модели

#### Поддерживаемые модели

* **HierBERTSegmentator** - наша основная модель для семантической сегментации русских новостных текстов. Состоит из двух моделей:
  * Sentence embedder - sentence_ru_cased_L-12_H-768_A-12_pt от DeepPavlov
  * BertForTextSegmentation - checkpoint-4000, обученная нами модель для сегментации текстов на основе эмбеддингов 
    от sentence_ru_cased_L-12_H-768_A-12_pt 

#### Обучение моделей

Ссылки на скрипты для подготовки данных, обучения и тестирования моделей расположены в директориях scripts соответствующих моделей

#### Результаты тестирования

Тестировали на датасете из 100 примеров (уточню позже), сделанных на основе корпуса Lenta.Ru

Считали метрику F1-score по каждому классу:

| Метрика       | HierBERTSegmentator |
|---------------|:-------------------:|
| "0" F1-score  |       0.9515        |
| "1" F1-score  |       0.6617        |
