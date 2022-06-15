FROM python:3.7-slim
MAINTAINER Aleksey Pauls <a.pauls@g.nsu.ru>

RUN python3 -m pip install --upgrade pip

# Copy code and models
COPY requirements.txt service.py load_models.py /usr/src/summarization_module/
COPY models/ /usr/src/summarization_module/models

WORKDIR /usr/src/summarization_module

RUN python3 -m pip install -r requirements.txt

# Оптимизируем установку PyTorch с помощью загруженного ранее whl (если такой файл существует)
RUN if [ -f "./models/torch-1.8.1+cpu-cp37-cp37m-linux_x86_64.whl" ] ; then \
    python3 -m pip install models/torch-1.8.1+cpu-cp37-cp37m-linux_x86_64.whl && \
    rm models/torch-1.8.1+cpu-cp37-cp37m-linux_x86_64.whl ; else \
    python3 -m pip install torch==1.8.1 ; fi

# Запкускаем скрипт, который загружает модели по необходимости
RUN python3 -u load_models.py

ENTRYPOINT ["python3", "service.py"]
