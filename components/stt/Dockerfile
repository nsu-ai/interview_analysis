FROM python:3.7.5-slim
MAINTAINER Daniil Grebenkin <d.grebenkin@g.nsu.ru>

RUN apt-get update && \
    yes | apt-get install apt-utils && \
    yes | apt-get install -y gcc && \
    yes | apt-get install -y make && \
    yes | apt-get install git g++ autoconf-archive make libtool && \
    yes | apt-get install python3-dev && \
    yes | apt-get install python-setuptools python-dev && \
    yes | apt-get install vim && \
    yes | apt-get install libsndfile1-dev && \
    yes | apt-get install sox && \
    yes | apt-get install libbz2-dev && \
    yes | apt-get install python-dev && \
    yes | apt-get install libxml2-dev && \
    yes | apt-get install libxslt-dev


COPY ./scripts/ /usr/src/scripts/
COPY ./models/ /usr/src/models/
WORKDIR /usr/src/scripts

RUN python3 --version
RUN export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.7.5/"

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install sox flask wave scipy pydub transformers pyctcdecode temp torch librosa numpy
RUN python3 -m pip install https://github.com/kpu/kenlm/archive/master.zip

RUN python3 load_models.py

ENTRYPOINT ["python3","core.py"]
