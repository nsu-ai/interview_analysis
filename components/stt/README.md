# speech2text

Это файлы модуля распознавания речи. Модуль может работать как на GPU, при наличии CUDA, так и на CPU. 
Если CUDA не установлен, то будет автоматически выбран CPU. 

Модуль использует 2 модели: 
1) предобученная wav2vec-модель (модель распознавания речи);
2) kenlm-модель, обученная на Common Crawl (языковая модель для рескоринга).

### Структура
    .
    ├── scripts                     # Необходимые для работы модуля скрипты 
    │   │ 
    │   └── core.py                 # Flask-app, главное приложение-контроллер модуля.
    │   │ 
    │   └── separator.py            # Скрипт, осуществляющий сепарацию длинных аудиозаписей (реализован также в виде библиотеки https://github.com/dangrebenkin/energy_separator)
    │   │ 
    │   └── augmentator.py          # Скрипт, который можно использовать для аугментации аудиозаписей различными шумами
    │   │ 
    │   └── load_models.py          # Скрипт, который осуществляет автозагрузку моделей в папку models
    │   
    ├── models                      # Реализации конкретных моделей
    │   │ 
    │   └── lm_commoncrawl.binary   # Бинарный файл kenlm-модели 
    │   │ 
    │   └── wav2vec2-large-ru-golos # Папка с файлами модели распознавания речи wav2vec для русского языка
    │
    ├── Dockerfile                  # Dockerfile для запуска модуля как сервиса
    ├── POST-request.py             # Тестовый файл с примером запроса для модуля
    ├── build                       # Файл, необходимый для сборки модуля    
    ├── test.wav                    # Тестовая аудиозапись для файла с запросом (POST-request.py)
    └── README.md (вы здесь)        # Описание модуля

### Cборка  и запуск сервиса

Перед сборкой необходимо скачать и поместить в папку 'models':
1) файл с языковой моделью "lm_commoncrawl.binary" (https://disk.yandex.ru/d/d2f5etV4aw2H2g); 
2) папку с моделью распознавания речи "wav2vec2-large-ru-golos" (https://disk.yandex.ru/d/U-r77za7CPQKVw).

Если их не загрузить вручную, то они скачаются автоматически при сборке докера.

Сборка докер-образа с сервисом производится командой
```
sudo docker build -t speech2text:0.3 .
```
Контейнер запускается командой
```
sudo docker run -p 127.0.0.1:8802:8802 speech2text:0.3
```
Файл 'POST-request.py' является примером, как можно передать контейнеру аудиофайл (желательно в формате wav и с частотой дискретизации 16000 кГц) 
и получить в ответ JSON-форматированный объект.

### Вызов сервиса

Для использования модуля как отдельного сервиса нужно отправлять
json файл на endpoint "/recognize" с помощью POST-запроса. Например, 
это можно сделать с помощью библиотеки requests и скрипта на python (см. файл 'POST-request.py'):
```python
import sys
import requests


def main():
	requests.get('http://localhost:8802/ready')  # проверяем готовность системы
	if len(sys.argv) > 1:
		sound_name = sys.argv[1]
	else:
		sound_name = 'test.wav'
	files = {'wav': (sound_name, open(sound_name, 'rb'), 'audio/wave')}
	resp = requests.post('http://localhost:8802/recognize', files=files)
	print(resp.json())


if __name__ == '__main__':
	main()

```
В качестве звукового файла (sound_name) для распознавания желательно отправлять файл формата WAV PCM, с частотой дискретизации 16000 Гц, с одним каналом (моно). При вызове сервиса необходимо указать его расположение (путь).

### Выходные данные

Объект json, содержащий поля:
1) timings, в который входят объекты json c полями 2), 3), 4);
2) word (распознанное слово);
3) start_time (временная отметка начала произнесения слова в момент в аудиозаписи);
4) end_time (временная отметка окончания произнесения слова в момент в аудиозаписи).

Пример:
``` 
{'timings': [{'end_time': 1.58, 'start_time': 1.12, 'word': 'мальчик'}, {'end_time': 2.14, 'start_time': 1.78, 'word': 'ворона'}]}
```

### Результаты тестов модели wav2vec2-large-ru-golos-v5

| Тестовые данные | Word Error Rate | Character Error Rate | Всего файлов | С рескорингом |
|---|---|---|---|---|
| Common Voice | 0.73 | 0.18 | 8007 | нет |
| Russian Open Speech To Text (asr_calls_2_val open_stt) | 0.69 | 0.37 | 13020 | да |
| Russian Open Speech To Text (buriy_audiobooks_2_val open_stt) | 0.43 | 0.14 | 7860 | да |
| Russian Open Speech To Text (public_youtube700_val open_stt) | 0.55 | 0.23 | 7338 | да |
| sberdevices_golos_tests (crowd) | 0.13 | 0.03 | 19994 | нет |
| sberdevices_golos_tests (farfield) | 0.26 | 0.07 | 3836 | нет |


