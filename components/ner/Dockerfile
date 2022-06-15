FROM python:3.7
MAINTAINER Ivan Bondarenko <i.bondarenko@g.nsu.ru>

# Install build utils
RUN apt-get update && \
    yes | apt-get install apt-utils && \
    yes | apt-get install -y gcc && \
    yes | apt-get install -y make && \
    yes | apt-get install -y apt-transport-https && \
    yes | apt-get install -y build-essential && \
    yes | apt-get install git g++ autoconf-archive make libtool && \
    yes | apt-get install python-setuptools python-dev && \
    yes | apt-get install python3-setuptools python3-dev

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Show versions
RUN python3 --version
RUN pip3 --version

# Create work directory and subdirectories
RUN mkdir /usr/src/named_entity_recognizer
RUN mkdir /usr/src/named_entity_recognizer/scripts
RUN mkdir /usr/src/named_entity_recognizer/models

# Copy code
COPY ./ner_service.py /usr/src/named_entity_recognizer/ner_service.py
COPY ./scripts/download_model.py /usr/src/named_entity_recognizer/scripts/download_model.py
COPY ./requirements.txt /usr/src/named_entity_recognizer/requirements.txt
COPY ./scripts/neural_network/ /usr/src/named_entity_recognizer/scripts/neural_network/
COPY ./scripts/data_processing/ /usr/src/named_entity_recognizer/scripts/data_processing/

# Set work directory
WORKDIR /usr/src/named_entity_recognizer

# Install all python dependencies
RUN python3 -m pip install -r requirements.txt

# Download model from Yandex Disk
RUN python3 scripts/download_model.py

ENTRYPOINT ["python3", "ner_service.py"]
