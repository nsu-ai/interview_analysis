# encoding: utf-8
# Version: 0.1

import os
import collections
from speechbrain.pretrained.interfaces import SepformerSeparation as separator
import torchaudio
import sox
import statistics
from vosk_speech_detection import detect_voice
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import wave



class Diarization:
    """ A class used to run audio-diarization process by solving the problem "who spoke and when".

    Process has two inner steps:
    First step: Sepformer (transformer-like) pre-trained pipeline separates audiofile into independent speaker's streams
    Second step: Voice Activity Detection (Vosk ASR model at the base) detects speech regions for every speaker by analyzing
    their files separately.

    Methods
    -------
    do_diarization: main method (speaker separation + voice activity detection)
    smooth: method to improve diarization accuracy
    convert: method to convert audio format (sample rate)
    """
    def __init__(self):
        # models initialization
        self.separator_pipeline = separator.from_hparams(source="speechbrain/sepformer-wsj02mix",
                                                         savedir='pretrained_models/sepformer-wsj02mix')
        SetLogLevel(0)
        self.model = Model('ru_vosk_model') # загружаем модель Kaldi/Vosk Ru

    def smooth(self, segments_dict):
        # сглаживание 
        previous = ()
        smoothed = collections.OrderedDict()
        first = True
        for seg_info in segments_dict.items():
            current_start, current_end, current_dur, current_label = '', '', '', ''
            current_start = "%.3f" % seg_info[0]
            current_end = seg_info[1][1]
            current_dur = seg_info[1][0]
            current_label = seg_info[1][2]
            if not first:
                prev_start, prev_end, prev_dur, prev_label = '', '', '', ''
                prev_start = previous[0]
                prev_end = previous[1][1]
                prev_dur = previous[1][0]
                prev_label = previous[1][2]
                # one label, one dictor
                if prev_label == current_label:
                    if float(current_start) >= float(prev_end):
                        diff = 0.0
                        diff = float(current_start) - float(prev_end)
                        if diff <= 0.3:
                            # same person -> merge two close segments (little pause)
                            start, end = '', ''
                            dur = 0.0
                            dur = float(prev_dur) + diff + float(current_dur)
                            del smoothed[prev_start]
                            smoothed[prev_start] = ("%.3f" % dur, current_end, current_label)
                            previous = (prev_start, ("%.3f" % dur, current_end, current_label))
                        else:
                            # same person -> long distance segments
                            smoothed[current_start] = (current_dur, current_end, current_label)
                            previous = (current_start, (current_dur, current_end, current_label))
                    elif float(prev_end) > float(current_start):
                        # same person -> merge two overlapping segments
                        start, end = '', ''
                        dur = 0.0
                        dur = float(end) - float(start)
                        del smoothed[prev_start]
                        smoothed[prev_start] = ("%.3f" % dur, current_end, current_label)
                        previous = (prev_start, ("%.3f" % dur, current_end, current_label))
                elif prev_label != current_label:
                    # different labels, different dictors
                    smoothed[current_start] = (current_dur, current_end, current_label)
                    previous = (current_start, (current_dur, current_end, current_label))
            elif first:
                smoothed[current_start] = (current_dur, current_end, current_label)
                previous = (current_start, (current_dur, current_end, current_label))
                first = False
        return smoothed

    def convert_file(self, path, target_rate):
        try:
            tfm = sox.Transformer()
            tfm.set_output_format(rate=target_rate)
            new_path = path[:-4] + '_' + str(target_rate) + '.wav'
            tfm.build_file(input_filepath=path, output_filepath=new_path)
            return new_path
        except Exception as e:
            print(e)

    def do_diarization(self, audiofile):
        dictor_parts_sources = None
        try:
            full_result = {}  # "dictor_i": [{} - VAD result, "filename.wav" - string with filename]
            dictor_parts_sources = self.separator_pipeline.separate_file(path=audiofile)
            dictors_number = 0
            dictors_number = len(dictor_parts_sources[0][0])
            print("dictors number after separation: " + str(dictors_number))
            for i in range(dictors_number):
                torchaudio.save("part_dictor_" + str(i) + ".wav", dictor_parts_sources[:, :, i].detach().cpu(), sample_rate=8000,
                                encoding="PCM_S", bits_per_sample=16)
                print('saved audio of dictor ' + str(i))
            all_dictors_vad_res = collections.OrderedDict()
            audiofiles_list = []
            for i in range(dictors_number):
                print("vosk detects for " + str(i))
                dictor_vad_result = []
                audio_for_vosk = None
                audio_for_vosk = self.convert_file("part_dictor_" + str(i) + ".wav", 16000)
                audiofiles_list.append(audio_for_vosk)
                dictor_vad_result = detect_voice(audio_for_vosk, self.model)
                os.remove("part_dictor_" + str(i) + ".wav")
                for speech_part in dictor_vad_result:
                    duration = 0.0
                    end = ''
                    start = ''
                    end = speech_part["end"]
                    start = speech_part["start"]
                    duration = float(end) - float(start)
                    all_dictors_vad_res[float(start)] = ("%.3f" % duration, end, str(i))
            smoothed_res = {}
            sorted_diar = collections.OrderedDict(sorted(all_dictors_vad_res.items()))
            smoothed_res = self.smooth(sorted_diar)
            return smoothed_res, audiofiles_list
        except Exception as e:
            print(e)
        finally:
            if dictor_parts_sources != None:
                for i in range(dictors_number):
                    if os.path.exists("part_dictor_" + str(i) + ".wav"):
                        os.remove("part_dictor_" + str(i) + ".wav")

