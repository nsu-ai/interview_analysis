FROM python:3.6-stretch
LABEL maintainer="Daria Nosenko dariacephei@gmail.com"

#ADD clean-layer.sh  /tmp/clean-layer.sh

RUN apt-get update && \
    yes |apt-get install -y gcc && \
    yes |apt-get install -y make && \
    yes |apt-get install -y apt-transport-https && \
    yes |apt-get install -y build-essential && \
    yes |apt-get install ffmpeg && \
    yes |apt-get install -y libsndfile1-dev


RUN python3 --version
RUN pip3 --version

COPY ./orchestrator.py /usr/src/main/orchestrator.py
COPY ./logging.ini /usr/src/main/logging.ini

WORKDIR /usr/src/main

RUN python -m pip install --upgrade pip

#RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install pydub
RUN pip3 install flask
RUN pip3 install requests

RUN mkdir -p /usr/src/main/log

#CMD /bin/sh
CMD ["python3", "orchestrator.py"]
