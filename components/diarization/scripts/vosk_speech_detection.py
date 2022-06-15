#!/usr/bin/env python3

import wave, json
from pydub import AudioSegment
import os
import sox
from vosk import KaldiRecognizer

# Vosk ASR-model for voice activity detection used
def detect_voice(path, kaldi_model):
    afile = None
    afile = wave.open(path, "rb")  # обработка входного файла
    all_words = [] 
    kaldi_recognizer = KaldiRecognizer(kaldi_model, 16000)
    kaldi_recognizer.SetWords(True)
    while True:
        data = None
        data = afile.readframes(4000)
        if len(data) == 0:
            break
        if kaldi_recognizer.AcceptWaveform(data):
            result = None
            result = json.loads(kaldi_recognizer.FinalResult())  # результат распознавания входного файла, json-объект
            if 'result' in result:
                for word in (result['result']):  # рассмотрим отдельно информацию для каждого распознанного слова
                    del word['conf']  # удаление confidence слов
                    end = ''
                    start = ''
                    end = '%.3f' % word['end']  # создание формата 3-х знаков после запятой для временной метки конца
                    word['end'] = end  # замена значения временной метки конца этим же значением в новом формате
                    start = '%.3f' % word['start']  # создание формата 3-х знаков после запятой для временной метки начала
                    word['start'] = start  # замена значения временной метки начала этим же значением в новом формате
                    all_words.append(word)
    return all_words
