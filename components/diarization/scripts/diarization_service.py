# encoding: utf-8
# Version: 0.1

import importlib
importlib.import_module('diarization_vosk')
from diarization_vosk import Diarization
import sox
import temp
import os
import wave
from flask import Flask, request, jsonify
import base64


app = Flask(__name__)

@app.route('/ready')
def ready():
    return 'OK'


@app.route('/recognize', methods=['POST'])
def recognize():
    """ Main method of diarization service to process input file and run model's analyzing process

    Parameters
    ----------
    audiofile: POST-request's body
        dictionary containing audiofile with speech

    Returns
    -------
    JSON with keys: 
    diarization:  (result of voice activity detection for source audio dictors)
    dictor_<number>_audio: result of speaker separation: unique audio file with dictor voice  
    """
    if 'wav' not in request.files:
        resp = jsonify({'message': 'No WAV part in the request'})
        resp.status_code = 400
        return resp
    audiofile_info = request.files['wav']
    input_file = audiofile_info.filename
    if input_file.endswith('.wav'):
        src_audiofile = temp.tempfile()+'.wav'
        audiofile_info.save(src_audiofile)
        rate = sox.file_info.sample_rate
        if rate != 8000:
            audiofile = temp.tempfile()+'.wav'
            tfm = sox.Transformer()
            tfm.set_output_format(rate=8000)
            tfm.build_file(input_filepath=src_audiofile, output_filepath=audiofile)
            os.remove(src_audiofile)
        else:
            audiofile = src_audiofile
        segments_list = []
        final_json_dict = {}
        diar_result = {}
        audiofiles = []
        try:
            # sending audio to Diarization class instance for processing  
            diar_result, audiofiles = diarization_model.do_diarization(audiofile)
            # parsing diarization result
            for dictor_data in diar_result.items():
                start, end, dictor = '', '', ''
                dictor = dictor_data[1][2]
                start = dictor_data[0]
                end = dictor_data[1][1]
                segments_list.append({"start_time": start, "end_time": end, "speaker": dictor})
            # filling final json
            final_json_dict["diarization"] = segments_list
            # parsing audiofiles
            for dictor_audio in audiofiles:
                label = ''
                encoded_audio = None
                raw_audio_str = ''
                label = dictor_audio[12:13]
                try:
                    encoded_audio = base64.b64encode(open(dictor_audio, "rb").read())
                    raw_audio_str = encoded_audio.decode('ascii')
                    os.remove(dictor_audio)
                except Exception as err:
                    print(err)
                # filling final json by audio bytes array
                final_json_dict['dictor_' + label + '_audio'] = raw_audio_str
            resp = jsonify(final_json_dict)
            return resp
        except Exception as e:
            resp = jsonify({'message': 'Failed processing file. Error: '+str(e)})
            os.remove(audiofile)
            resp.status_code = 400
        finally:
            home = '/usr/src/main'
            if os.path.exists(audiofile):
                os.remove(audiofile)
            if os.path.exists(os.path.join(home, audiofile[5:])):
                os.remove(os.path.join(home, audiofile[5:]))
            for filek in os.listdir(home):
                if filek.endswith(".wav"):
                    os.remove(os.path.join(home, filek))
    else:
        resp = jsonify({'message': 'Wrong extension of audiofile!'})
        resp.status_code = 400
        return resp


if __name__ == '__main__':
    diarization_model = Diarization()
    app.run(host='0.0.0.0', port=8801)
