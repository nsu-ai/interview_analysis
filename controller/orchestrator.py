# encoding: utf-8
# Version: 0.1

from flask import Flask, request, jsonify
import requests
import codecs
import copy
import json
import logging
import tempfile
from logging.config import fileConfig
import os
from os import path
import time
from pydub import AudioSegment
from pydub.utils import mediainfo
import sys
import base64
import io


fileConfig(path.join(path.dirname(path.abspath(__file__)), 'logging.ini'))
logger = logging.getLogger('main_logger')


class Chief(object):
    """ A class used to process audio file recognition by managing work of sub-web-modules

        Attributes
        ----------
        services urls
        jsons for results of services
        available: `list`
            - list of available services
        steps for correct execution order
        json for final system's result

        Methods
        -------
        check_ready_status: static method for 'ping'-like services simple healthcheck + defining of available services
        healthcheck: healthcheck for services containers building
        check_service: sub-method for check_ready_status
        process_file: main class method to orchestrate services work
        """ 

    def __init__(self, diar_url, stt_url, punct_url, ner_url, sent_url, seg_url, summ_url):
        self.ready_diar = False
        self.ready_stt = False
        self.ready_punct = False
        self.ready_ner = False
        self.ready_sent = False
        self.ready_seg = False
        self.ready_summ = False
        self.available = []
        self.step0 = False
        self.step1 = False
        self.step2 = False
        self.step3 = False
        self.step4 = False
        self.step5 = False
        self.stt_error_occured = False
        self.diar_json = ''
        self.stt_jsons = {}
        self.punct_json = ''
        self.ner_json = ''
        self.sent_json = ''
        self.seg_json = ''
        self.summ_json = ''
        self.combined_diar_stt_list = []
        self.diar_url = diar_url
        self.stt_url = stt_url
        self.punct_url = punct_url
        self.ner_url = ner_url
        self.sent_url = sent_url
        self.seg_url = seg_url
        self.summ_url = summ_url
        self.result_json = ''
        self.work_res = None
        self.audiofiles = {}


    @staticmethod
    def check_ready_status(service_url):
        try:
            logger.debug('checking ' + service_url + '...')
            resp = requests.get(service_url + '/ready')
            if resp.status_code == 200:
                logger.debug(service_url + ' is ready!')
                return True
            else:
                logger.debug(service_url + " isn't ready for work")
                return False
        except:
            logger.debug(service_url + " isn't ready for work")
            return False

    @staticmethod
    def decode_audiofile(coded_bytes):
        decoded_bytes = coded_bytes.encode("ascii")
        audio_decoded_string = base64.b64decode(decoded_bytes)
        return audio_decoded_string	

    def easy_wait(self, serv_url):
        counter = 0
        a = False
        while counter != 4:
            serv_r = self.check_ready_status(serv_url)
            if not serv_r:
                time.sleep(20)
                counter += 1
            else:
                a = True
                break
        return a

    def healthcheck(self):
        avail = []
        serv_keys = {'diar': self.diar_url, 'stt': self.stt_url, 'punct': self.punct_url, 'ner': self.ner_url, 'sent': self.sent_url, 'seg': self.seg_url, 'summ': self.summ_url}
        for s_pair in serv_keys.items():
            serv_resp = self.easy_wait(s_pair[1])
            if serv_resp:
                avail.append(s_pair[0])
        self.available = avail

    def combine_diar_stt_dict(self, dictor):
        filled_segments = []
        diar_dict = {}
        stt_dict = {}
        diar_segments = self.combined_diar_stt_list
        print('dair segments : ')
        print(diar_segments)
        print('stt dict ')
        stt_dict = self.stt_jsons[dictor]
        print(stt_dict)
        for segment in diar_segments:
            seg_start = float(segment["start_time"])
            seg_end = float(segment["end_time"])
            seg_label = segment["speaker"]
            if seg_label == dictor[7:8]:
                if "words" not in segment.keys():
                    segment_words = []
                    for word_dict in stt_dict["timings"]:
                        word_start = float(word_dict["start_time"])
                        word_end = float(word_dict["end_time"])
                        if word_start >= seg_start and word_end <= seg_end:
                            segment_words.append(word_dict)
                    if len(segment_words) != 0:
                        filled_segments.append({"start_time": seg_start, "end_time": seg_end, "speaker": seg_label, "words": segment_words})
                    else:
                        filled_segments.append({"start_time": seg_start, "end_time": seg_end, "speaker": seg_label})
            elif seg_label != dictor[7:8]:
                filled_segments.append({"start_time": seg_start, "end_time": seg_end, "speaker": seg_label})
        self.combined_diar_stt_list = filled_segments

    def process_file(self, filepath, options=None):
        """ A method used to run file recognition process

            Parameters
            ----------
            filepath: `str`
                path to audio file containing speech
            (optional) options: `List`
                list of services names to apply

            Returns
            -------
            analysing_result: `JSON`
            """
        try:
            logger.info('~~~Processing file~~~')
            AUDIO_DATA = {}
            AUDIO_DATA = {'wav': (filepath, open(filepath, 'rb'), 'audio/wave')}
            if options is not None:
                for val in options:
                    val = 'ready_' + val
                    self.__setattr__(val, True)
            else:
                for it in self.available:
                    logger.info(it)
                    call = 'ready_' + it
                    self.__setattr__(call, True)
            if self.ready_diar:
                wait_status = True
                while wait_status:
                    if not self.check_ready_status(self.diar_url):
                        logger.debug('sleep diar')
                        time.sleep(5)
                        sys.exit()
                    elif self.check_ready_status(self.diar_url):
                        logger.info('~~~Running diarization~~~')
                        diar_response = requests.post(self.diar_url + '/recognize', files=AUDIO_DATA)
                        logger.debug('Diarization service//response status code ' + str(diar_response.status_code))
                        logger.info(diar_response.json().keys())
                        #diar_response.encoding = 'UTF-8'
                        dr_resp = diar_response.json()
                        if "diarization" in dr_resp:
                            self.combined_diar_stt_list = dr_resp["diarization"]
                            for any_key in dr_resp.keys():
                                if "dictor" in any_key:
                                    decoded_audio_string = self.decode_audiofile(dr_resp[any_key])
                                    self.audiofiles[any_key] = decoded_audio_string
                            self.step0 = True
                            self.diar_json = copy.deepcopy(diar_response.json())
                            diar_response.close()
                            logger.info('~~~Diarization finished, OK~~~')
                            logger.info('___Diarization result___')
                            logger.info(self.diar_json["diarization"])
                            logger.info('___End___')
                            wait_status = False
                        elif "diarization" not in dr_resp:
                            logger.info('Error with diarization process.')
                            logger.info('~~~We cannot process and analyse your audiofile, speech recognition module needs dictor files from diarization module! Modules (punctuation restoration, NER, semantic segmentation, sentiment analysis, summarization) need speech recognition module result ! ! ! ~~~')
                            self.step0 = False
                            logger.debug(dr_resp)
                            diar_response.close()
                            wait_status = False
            elif not self.ready_diar:
                self.step0 = False
                logger.info('~~~We cannot process and analyse your audiofile, speech recognition module needs dictor files from diarization module! Modules (punctuation restoration, NER, semantic segmentation, sentiment analysis, summarization) need speech recognition module result ! ! ! ~~~')
            if self.step0:
                if self.ready_stt:
                    wait_status = True
                    while wait_status:
                        if not self.check_ready_status(self.stt_url):
                            time.sleep(5)
                            sys.exit()
                        elif self.check_ready_status(self.stt_url):
                            logger.info('~~~Running speech recognition~~~')
                            for dictor_audio in self.audiofiles.items():
                                dictor_name = ''
                                dictor_audio_stream = dictor_audio[1]
                                dictor_name = dictor_audio[0]
                                with open(dictor_name+'.wav', 'wb') as to_out:
                                    to_out.write(dictor_audio_stream)
                                AudioSegment.from_file(io.BytesIO(dictor_audio_stream)).export(dictor_name + '.wav', format='wav')
                                STT_DATA = {'wav': (dictor_name+'.wav', open(dictor_name + '.wav', 'rb'), 'audio/wave')}
                                #STT_DATA = {'wav': (dictor_name+'.wav', open(dictor_audio_stream, 'rb'), 'audio/wave')}
                                stt_response = requests.post(self.stt_url + '/recognize', files=STT_DATA)
                                logger.debug('Speech recognition service//response status code ' + str(stt_response.status_code))
                                stt_response.encoding = 'UTF-8'
                                stt_r = stt_response.json()
                                print('stt json')
                                print(stt_r)
                                if "timings" not in stt_r:
                                    logger.info('Error in speech recognition process.')
                                    logger.info('Other modules (punctuation restoration, NER, semantic segmentation, sentiment analysis, summarization) need STT result. Passing with diarization result only. ')
                                    logger.debug(stt_r)
                                    stt_response.close()
                                    self.stt_error_occured = True
                                    break
                                elif "timings" in stt_r:
                                    self.stt_jsons[dictor_name] = stt_r
                                    print('saved for dictor : ' + dictor_name)
                                #self.stt_jsons[dictor_name] = copy.deepcopy(stt_response.json())
                                    stt_response.close()
                                    self.combine_diar_stt_dict(dictor_name) #  update diar_stt list by new dictor words
                                    print('combined diar stt for ' + dictor_name)
                                    print(self.combined_diar_stt_list)
                                    if os.path.exists(dictor_name+'.wav'):
                                        os.remove(dictor_name+'.wav')
                            #self.combined_diar_stt_json = json.dumps(combined_diar_stt_dict)
                            #with codecs.open(self.stt_json, mode='w', encoding='utf-8', errors='ignore') as stt_f:
                                #json.dump(stt_json, fp=stt_f, ensure_ascii=False)
                            if not self.stt_error_occured:   
                                self.step1 = True
                                logger.info('~~~Speech recognition finished, OK')
                                logger.info('___Diarization segments with words from Speech-to-text module___')
                                logger.info(self.combined_diar_stt_list)
                                logger.info('___End___')
                                wait_status = False
                            elif self.stt_error_occured:
                                self.step1 = False
                                wait_status = False
                elif not self.ready_stt:
                    self.step1 = False
                    logger.info('~~~Pass without speech recognition and other modules (punctuation restoration, NER, semantic segmentation, sentiment analysis, summarization). All other modules need STT result ! ! ! ~~~')
            if self.step1:
                if self.ready_punct:
                    wait_status = True
                    while wait_status:
                        if not self.check_ready_status(self.punct_url):
                            time.sleep(5)
                        if self.check_ready_status(self.punct_url):
                            logger.info('~~~Running punctuation restoration~~~')
                            punct_response = requests.post(self.punct_url + '/recognize', data=json.dumps({"diarization": self.combined_diar_stt_list}), headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
                            punct_response.encoding = 'UTF-8'
                            punct_r = punct_response.json()
                            logger.debug(
                                'Punctuation restoration service//response status code ' + str(punct_response.status_code))
                            if "text" in punct_r:
                                logger.info('~~~Punctuation restoration finished, OK')
                                logger.info('___Added Punctuation restoration result___')
                                logger.info(punct_r)
                                logger.info('___End___')
                                self.punct_json = copy.deepcopy(punct_r)
                                punct_response.close()
                                wait_status = False
                                self.step2 = True
                            elif "text" not in punct_r:
                                logger.info('Error with punctuation restoration')
                                logger.info('Pass without punctuation restoration and other modules (NER, semantic segmentation, sentiment analysis, summarization). All other modules need punctuation restoration result.')
                                logger.debug(punct_r)
                                punct_response.close()
                                wait_status = False
                                self.step2 = False
                elif not self.ready_punct:
                    self.step2 = False
                    logger.info('~~~Pass without punctuation restoration and other modules (NER, semantic segmentation, sentiment analysis, summarization). All other modules need punctuation restoration result ! ! ! ~~~')
            if self.step2:
                if self.ready_ner:
                    wait_status = True
                    while wait_status:
                        if not self.check_ready_status(self.ner_url):
                            time.sleep(5)
                        if self.check_ready_status(self.ner_url):
                            logger.info('~~~Running NER~~~')
                            punct_data = self.punct_json["text"]
                            data_for_ner = {"result": [{"segment": self.combined_diar_stt_list, "text": punct_data}]}
                            logger.debug('data for ner ')
                            logger.debug(data_for_ner)
                            logger.debug(type(data_for_ner))
                            ner_response = requests.post(self.ner_url + '/recognize', data=json.dumps(data_for_ner),
                                                         headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
                            ner_response.encoding = 'UTF-8'
                            ner_r = ner_response.json()
                            logger.debug('NER service//response status code ' + str(ner_response.status_code))
                            if "ners" in ner_r["result"][0]:
                                logger.info('~~~NER finished, OK~~~')
                                logger.info('___Added NER result___')
                                logger.info(ner_r["result"][0])
                                logger.info('___End___')
                                self.ner_json = copy.deepcopy(ner_r)
                                ner_response.close()
                                wait_status = False
                                self.step3 = True
                            elif "ners" not in ner_r["result"][0]:
                                logger.info("Error in NER process.")
                                logger.info("Pass without enitities from NER module...")
                                logger.debug(ner_r)
                                ner_response.close()
                                wait_status = False
                                self.step3 = True
                elif not self.ready_ner:
                    self.step3 = True
                    logger.info('~~~Pass without NER ~~~')
            if self.step3:
                if self.ready_seg:
                    wait_status = True
                    while wait_status:
                        if not self.check_ready_status(self.seg_url):
                            time.sleep(5)
                        if self.check_ready_status(self.seg_url):
                            logger.info('~~~Running Semantic Segmentation~~~')
                            logger.debug('PUNCT JSON')
                            logger.debug(self.punct_json)
                            seg_response = requests.post(self.seg_url + '/recognize', data=json.dumps(self.punct_json),
                                                         headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
                            seg_response.encoding = 'UTF-8'
                            seg_r = seg_response.json()
                            logger.debug('Semantic segmentation service//response status code ' + str(seg_response.status_code))
                            if "segments" in seg_r:
                                logger.info('~~~Semantic segmentation finished, OK~~~')
                                logger.info('___Added Semantic segmentation result___')
                                logger.info(seg_r)
                                logger.info('___End___')
                                self.seg_json = copy.deepcopy(seg_r)
                                seg_response.close()
                                wait_status = False
                                self.step4 = True
                            elif "segments" not in seg_r:
                                logger.info("Error in semantic segmentation process.")
                                logger.info("Pass without segments ...")
                                logger.debug(seg_r)
                                seg_response.close()
                                wait_status = False
                                self.step4 = True
                        elif not self.ready_seg:
                            self.step4 = True
                            logger.info('~~~Pass without Semantic segmentation~~~')
            if self.step4:
                if self.ready_summ:
                    wait_status = True
                    while wait_status:
                        if not self.check_ready_status(self.summ_url):
                            time.sleep(5)
                        if self.check_ready_status(self.summ_url):
                            logger.info('~~~Running Summarization~~~')
                            data_for_summ = ''
                            if self.seg_json != '':
                                data_for_sum = self.seg_json
                            elif self.seg_json == '':
                                data_for_sum = self.punct_json
                            sum_response = requests.post(self.summ_url + '/recognize', data=json.dumps(data_for_sum),
                                                         headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
                            sum_response.encoding = 'UTF-8'
                            sum_r = sum_response.json()
                            logger.debug('Summarization service//response status code ' + str(sum_response.status_code))
                            if "summaries" in sum_r:
                                logger.info('~~~Summarization finished, OK~~~')
                                logger.info('___Added Summarization result___')
                                logger.info(sum_r)
                                logger.info('___End___')
                                self.summ_json = copy.deepcopy(sum_r)
                                sum_response.close()
                                wait_status = False
                                self.step5 = True
                            elif "summaries" not in sum_r:
                                logger.info("Error in summarization process.")
                                logger.info("Pass without summaries ...")
                                logger.debug(sum_r)
                                sum_response.close()
                                wait_status = False
                                self.step5 = True
                        elif not self.ready_summ:
                            self.step5 = True
                            logger.info('~~~Pass without Summarization~~~')
            if self.step5:
                if self.ready_sent:
                    wait_status = True
                    while wait_status:
                        if not self.check_ready_status(self.sent_url):
                            time.sleep(5)
                        if self.check_ready_status(self.sent_url):
                            logger.info('~~~Running Sentiment Analysis~~~')
                            data_for_sent = ''
                            if self.summ_json != '' and self.seg_json != '':
                                data_for_sent = self.summ_json
                            elif self.summ_json == '' and self.seg_json != '':
                                data_for_sent = self.seg_json
                            else:
                                data_for_sent = self.punct_json
                            sent_response = requests.post(self.sent_url + '/recognize', data=json.dumps(data_for_sent),
                                                         headers={'Content-type': 'application/json', 'Accept': 'text/plain'})
                            sent_response.encoding = 'UTF-8'
                            sent_r = sent_response.json()
                            logger.debug('Sentiment analysis service//response status code ' + str(sent_response.status_code))
                            if "sentiments" in sent_r:
                                logger.info('~~~Sentiment analysis finished, OK~~~')
                                logger.info('___Added sentiment analysis result___')
                                logger.info(sent_r)
                                logger.info('___End___')
                                self.sent_json = copy.deepcopy(sent_r)
                                sent_response.close()
                                wait_status = False
                            elif "sentiments" not in sent_r:
                                logger.info("Error in sentiment analysis process.")
                                logger.info("Pass without sentiments ...")
                                logger.debug(sent_r)
                                sent_response.close()
                                wait_status = False
                elif not self.ready_sent:
                    logger.info('~~~Pass without Sentiment Analysis ~~~')
            merged_res = {}
            if self.diar_json == '':
                error = 'We cannot process and analyse your audiofile, speech recognition module needs dictor files from diarization module! Modules (punctuation restoration, NER, semantic segmentation, sentiment analysis, summarization) need speech recognition module result ! ! !'
                logger.error(error)
                return error, 1
            else:
                diar_dict = {}
                punct_dict = {}
                ner_dict = {}
                seg_dict = {}
                sum_dict = {}
                sent_dict = {}
                if self.diar_json != '' and len(self.stt_jsons.items()) != 0:
                    merged_res["result"] = self.combined_diar_stt_list
                elif self.diar_json != '' and len(self.stt_jsons.items()) == 0:
                    diar_dict = self.diar_json
                    merged_res.update(diar_dict)
                if self.punct_json != '':
                    punct_dict = self.punct_json
                    merged_res.update({"text": punct_dict["text"]})
                if self.ner_json != '':
                    ner_dict = self.ner_json
                    merged_res.update({"entities": ner_dict["result"][0]["ners"]})
                if self.seg_json != '':
                    seg_dict = self.seg_json
                    merged_res.update({"segments": seg_dict["segments"]})
                if self.summ_json != '':
                    sum_dict = self.summ_json
                    merged_res.update({"summaries": sum_dict["summaries"]})
                if self.sent_json != '':
                    sent_dict = self.sent_json
                    merged_res.update({"sentiments": sent_dict["sentiments"]})
                if merged_res is not None:
                    # self.result = get_temp_name('.json')
                    # with codecs.open(self.result, mode='w', encoding='utf-8', errors='ignore') as resj:
                    #     merged_res = json.dumps(merged_res, fp=resj, ensure_ascii=False)
                    self.work_res = merged_res
                else:
                    err = "Error: Empty result"
                    logger.error(err)
                    return err, 1
        except Exception as e:
            err = "Error: {}".format(e)
            logger.error(err)
            return err, 1
        finally:
            pass
            #try:
                #os.remove(self.json)
            #except Exception as e:
                #pass
            #finally:
                #pass
        self.step0 = False
        self.step1 = False
        self.step2 = False
        self.step3 = False
        self.step4 = False
        self.step5 = False
        self.diar_json = ''
        self.punct_json = ''
        self.ner_json = ''
        self.seg_json = ''
        self.sent_json = ''
        self.summ_json = ''
        self.stt_error_occured = False
        self.audiofiles = {}
        self.stt_jsons = {}
        self.combined_diar_stt_list = []
        self.ready_diar = False
        self.ready_stt = False
        self.ready_punct = False
        self.ready_ner = False
        self.ready_sent = False
        self.ready_seg = False
        self.ready_summ = False
        return self.work_res, 0


app = Flask(__name__)
diar = 'http://diar-service:8801'
stt = 'http://stt-service:8802'
punct = 'http://punct-service:8803'
ner = 'http://ner-service:8977'
sent = 'http://sent-service:8805'
seg = 'http://seg-service:8806'
summ = 'http://sum-service:8807'


def get_temp_name(key):
    fp = tempfile.NamedTemporaryFile(delete=True, suffix=key)
    file_name = fp.name
    fp.close()
    del fp
    return file_name


@app.route('/ready')
def ready():
    """ This method looks for orchestrator instance's list of available services and returns to user info about
    available for current session options to process and analyze audiofile.

    Parameters
    ----------
    audiofile: POST-request's body
        dictionary containing audiofile with speech

    Returns
    -------
    diarization_result: `JSON`
    """
    services = 'Available options: '
    if controller.available != None:
        opts = controller.available
        logger.info(opts)
        for key in opts:
            if key == 'diar':
                services += key + ' (to run diarization). '
            elif key == 'stt':
                services += key + ' (to run speech recognition). '
            elif key == 'punct':
                 services += key + ' (to run punctuation restoration). '
            elif key == 'ner':
                 services += key + ' (to run named entity recognition). '
            elif key == 'sent':
                 services += key + ' (to run sentiment analysis). '
            elif key == 'seg':
                services += key + ' (to run segmentation). '
            elif key == 'summ':
                services += key + ' (to run summarization) '
        logger.info('~~~System is ready to analyse file~~~')
        logger.info(services)
        resp = jsonify({'message': 'System is ready to analyse file. ' + services})
        resp.status_code = 200
        return resp
    else:
        resp = jsonify({'message': "System isn't ready. No available services"})
        resp.status_code = 400
        logger.info("System isn't ready. No available services")
        return resp


ALLOWED_FREQ = 8000
#ALLOWED_BITDEPTH = 'PCM_16'
ALLOWED_NUM_OF_CHANNELS = 1
ALLOWED_DURATION = 500.0


@app.route('/recognize', methods=['POST'])
def recognize():
    """ Main method of orchestrator service to preprocess input audio file and transfer it to services.

    Parameters
    ----------
    request.files: POST-request's body containing:
        - dictionary containing audiofile with speech
        - (optional) options `string` to specify concrete services for file processing

    Returns
    -------
    resp: `JSON`
        structured file processing result representing the concatenation of services json's
    """
    logger.info('~~~Recognition process started~~~')

    if 'audio' not in request.files:
        logger.error('400: No audiofile part in the request')
        resp = jsonify({'message' : 'No audiofile part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['audio']

    if file.filename == '':
        logger.error('400: No audio file provided for upload')
        resp = jsonify({'message': 'No audio file provided for upload'})
        resp.status_code = 400
        return resp

    if 'options' not in request.files:
        logger.info('Pass without options, using all available services.')
        opts = None
    else:
        logger.info('Process with options')
        opts_dict = json.loads(request.files['options'])
        opts = opts_dict['options'].split('/')

    source_audio_extension = path.splitext(file.filename)[1].replace('.',  '')
    logger.debug('extension ' + source_audio_extension)
    if file:
        secret_name = get_temp_name('.' + source_audio_extension)
        file.save(secret_name)
        logger.info('Saved audio file with tmp name')
        logger.debug('name '+secret_name)
        sound = None
        convert = False
        rate = False
        channel = False
        logger.info('Started cheking audiofile')
        try:
            if source_audio_extension == 'wav':
                sound = AudioSegment.from_wav(secret_name)
            elif source_audio_extension == 'mp3':
                sound = AudioSegment.from_mp3(secret_name)
                logger.debug('checked mp3')
                convert = True
            else:
                sound = AudioSegment.from_file(secret_name, source_audio_extension)
                convert = True
            num_ch = sound.channels
            num_fr = sound.frame_rate
            if num_fr != 8000:
                rate = True
                logger.info('Converting file frequency from ' + str(num_fr) + ' Hz to 8000 Hz')
                sound.set_frame_rate(8000)
            if num_ch != 1:
                channel = True
                logger.info('Converting file channels from ' + str(num_ch) + ' to mono')
                sound.set_channels(1)
            if convert:
                logger.info('Converting file format from ' + source_audio_extension + ' to WAV')
                export_name = secret_name.split('.')[0] + '.wav'
                sound.export(export_name, format="wav")
                secret_name = export_name
                #os.remove(secret_name)
            if not convert:
                if rate or channel:
                    sound.export(secret_name, format="wav")
            logger.info('File checked and ready to be analyzed')
        except Exception as e:
            resp = jsonify({'message' : 'Unknown input format! Supportable formats listed in ffmpeg docs'})
            logger.debug(str(e))
            if path.exists(secret_name):
                os.remove(secret_name)
            resp.status_code = 400
            return resp
        try:
            result, err = controller.process_file(secret_name, opts)
        except Exception as e:
            err = 1
            result = "Error: {}".format(e)

        if err == 1:
            #resp = jsonify({"message": result})
            logger.info(result)
            logger.error('400: Recognition process failed')
            resp = jsonify({"message": "Sorry. We can't process your file. Some error occured during recognition process. "})
            resp.status_code = 400
            if path.exists(secret_name):
                os.remove(secret_name)
            return resp
        resp = jsonify({"result": result})
        logger.info('Result: ' + json.dumps(result))
        resp.status_code = 201
        if path.exists(secret_name):
            os.remove(secret_name)
        return resp


if __name__ == '__main__':
    controller = Chief(diar, stt, punct, ner, sent, seg, summ)
    controller.healthcheck()
    logger.debug('Created system')
    app.run(host='0.0.0.0', port=8800)
