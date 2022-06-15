# -*- coding: utf-8 -*-
from tempfile import NamedTemporaryFile
import os, sox, json, wave, warnings, numpy
from flask import Flask, request, jsonify
from separator import separate

#--------------------
asr_model = '../models/wav2vec2-large-ru-golos'
language_model = '../models/lm_commoncrawl.binary'

from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from pyctcdecode import build_ctcdecoder
import librosa, torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained(asr_model)
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
decoder = build_ctcdecoder(
	labels=list(sorted_vocab_dict.keys()),
	kenlm_model_path=language_model,
)
feature_extractor = Wav2Vec2FeatureExtractor(
		feature_size=1,
		sampling_rate=16000,
		padding_value=0.0,
		do_normalize=True,
		return_attention_mask=True)
tokenizer_2 = Wav2Vec2CTCTokenizer.from_pretrained(asr_model)
processor_with_lm = Wav2Vec2ProcessorWithLM(
	feature_extractor=processor.feature_extractor,
	tokenizer=tokenizer_2,
	decoder=decoder
)
model = Wav2Vec2ForCTC.from_pretrained(asr_model).to(device)
#--------------------

def speech_recognition(audiofiles):
	
	global processor, model, processor_with_lm, feature_extractor
	
	transcriptions = []
	vectors = []
	for audiofile in audiofiles:
		speech_array, sampling_rate = librosa.load(audiofile, sr=16_000)
		os.remove(audiofile)
		vectors.append(speech_array)
	inputs = processor(vectors, sampling_rate=16_000, return_tensors="pt", padding=True).to(device)
	with torch.no_grad():
		logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
		
	if len(logits) >= 2:
		final_logits = logits[0].cpu().numpy()
		for logit in logits[1::]:	
			logit = logit.cpu().numpy()
			final_logits = numpy.concatenate((final_logits, logit), axis=0)
		final_logits = decoder.decode_beams(final_logits)[0]
		transcript, lm_state, indices, logit_score, lm_score = final_logits
		time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
		for file_offset in indices:
			word_offset = {
								"word": file_offset[0],
								"start_time": round(float(file_offset[1][0]) * time_offset, 2), # в секундах
								"end_time": round(float(file_offset[1][1]) * time_offset, 2), # в секундах
							}
							
			transcriptions.append(word_offset)
	else:
		final_logits = logits.cpu().numpy()
		results = processor_with_lm.batch_decode(logits=final_logits, output_word_offsets = True)
		time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate
		for file_offset in results.word_offsets:
			word_offsets = [
							{
								"word": d["word"],
								"start_time": round(d["start_offset"] * time_offset, 2), # в секундах
								"end_time": round(d["end_offset"] * time_offset, 2), # в секундах
							}
							for d in file_offset]
			for offset in word_offsets:
				transcriptions.append(offset)
	return transcriptions

def preprocessor(path):
	global asr
	seconds = sox.file_info.duration(path)
	if seconds >= 120.0:
		audio_path = separate(path)
		offsets = speech_recognition(audio_path)			
		string = json.dumps(dict(timings = offsets), ensure_ascii=False)
	if seconds < 120.0:
		audio_path = [path]
		offsets = speech_recognition(audio_path)
		string = json.dumps(dict(timings = offsets), ensure_ascii=False)
	result = json.loads(string)
	return result

app = Flask(__name__)

@app.route('/ready')
def ready():
    return 'OK'

@app.route('/recognize', methods=['POST'])
def recognize():
	if 'wav' not in request.files: # проверка наличия wav-файла в входных данных
		resp = jsonify({'message': 'No WAV part in the request'})
		resp.status_code = 400
		return resp
	audiofile_info = request.files['wav']
	input_file = audiofile_info.filename
	if input_file.endswith('.wav'): # проверка расширения входного файла
		src_audiofile = ''
		audiofile = ''
		try:
			try:
				with NamedTemporaryFile(suffix='.wav', delete=False) as src_fp:
					src_audiofile = src_fp.name
				audiofile_info.save(src_audiofile)
				audiofile_params = sox.file_info.info(src_audiofile)
				sample_rate = audiofile_params['sample_rate']
				channels = audiofile_params['channels']
				bitdepth = audiofile_params['bitdepth']
				encoding = audiofile_params['encoding']
				if (channels == 1) and (sample_rate == 16000) and (bitdepth == 16) and (encoding == 'signed-integer'): # проверка параметров аудитофайла (должно быть моно, 8 бит, 16кГц)
					audiofile = src_audiofile
				else:
					if channels != 1:
						warnings.warn(f'Channels: expected 1, got {channels}.')
					if sample_rate != 16000:
						warnings.warn(f'Sample rate: expected 16000, got {sample_rate}.')
					if bitdepth != 16:
						warnings.warn(f'Bits per sample: expected 16, got {bitdepth}.')
					if encoding != 'signed-integer':
						warnings.warn(f'Bits per sample: expected signed-integer, got {encoding}.')
					with NamedTemporaryFile(suffix='.wav', delete=False) as dst_fp:
						audiofile = dst_fp.name
					tfm = sox.Transformer()
					tfm.set_output_format(rate=16000, channels=1, bits=16, encoding='signed-integer')
					tfm.build_file(
						input_filepath=src_audiofile,
						output_filepath=audiofile
					)
				err_msg = ''
			except BaseException as err:
				err_msg = str(err)
			if (len(err_msg) > 0) or (len(audiofile) == 0):
				resp = jsonify({'message': f'Wrong audiofile! {err_msg}'})
				resp.status_code = 400
			else:
				result_asr = preprocessor(audiofile) # вызов функции распознавания речи
				resp = jsonify(result_asr)
		finally:
			if os.path.isfile(src_audiofile):
				os.remove(src_audiofile)
			if os.path.isfile(audiofile):
				os.remove(audiofile)
	else:
		resp = jsonify({'message': 'Wrong extension of audiofile!'})
		resp.status_code = 400
	return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8802)
