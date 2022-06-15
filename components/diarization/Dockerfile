FROM python:3.7-slim-buster
LABEL maintainer="Daria Nosenko dariacephei@gmail.com"

#RUN RUN chmod +x /tmp/clean-layer.sh

#ADD clean-layer.sh  /tmp/clean-layer.sh

RUN apt-get update && \
    yes |apt-get install -y --no-install-recommends apt-utils && \
    yes |apt-get install -y gcc && \
    yes |apt-get install -y make && \
    yes |apt-get install -y apt-transport-https && \
    yes |apt-get install -y build-essential && \
    yes |apt-get install sox && \
    yes |apt-get install ffmpeg && \
    yes |apt-get install -y libsndfile1-dev \
    yes |apt-get install libtinfo5
    #yes |apt-get install -y --no-install-recommends apt-utils

RUN python3 --version
RUN pip3 --version
RUN python -m pip install --upgrade pip

COPY ./scripts/diarization_service.py /usr/src/main/diarization_service.py
COPY ./scripts/diarization_vosk.py /usr/src/main/diarization_vosk.py
COPY ./scripts/vosk_speech_detection.py /usr/src/main/vosk_speech_detection.py
COPY ./scripts/load_models.py /usr/src/main/load_models.py
#COPY ./models/ru_vosk_model/ /usr/src/main/ru_vosk_model/
COPY ./requirements.txt /usr/src/main/requirements.txt
#COPY torch-1.8.1+cpu-cp37-cp37m-linux_x86_64.whl /usr/src/main/torch-1.8.1+cpu-cp37-cp37m-linux_x86_64.whl

WORKDIR /usr/src/main

#RUN RUN chmod +x /tmp/clean-layer.sh

#RUN pip3 install torch>=1.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install --no-cache-dir --default-timeout=900 torch==1.11.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torch-1.8.1+cpu-cp37-cp37m-linux_x86_64.whl
RUN pip3 install speechbrain
RUN pip3 install flask
RUN pip3 install argparse sox flask temp wave scipy pydub
RUN pip3 install vosk==0.3.32
RUN pip3 install soxbindings

RUN python3 load_models.py

CMD ["python3", "diarization_service.py"]
#CMD /bin/sh
