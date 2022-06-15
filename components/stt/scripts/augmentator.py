import os, zipfile, random, logging, traceback
from pydub import AudioSegment

class Augmentator:

	logging.basicConfig(filename='augmentator.log', level=logging.INFO)

	noise_decibels = 10.0
	folder_to_outputs = os.path.abspath(os.getcwd())
	sample_rate = 16000

	def __init__(self, noise_decibels=noise_decibels, folder_to_outputs=folder_to_outputs, sample_rate=sample_rate):
		self.noise_decibels = noise_decibels
		self.folder_to_outputs = folder_to_outputs
		self.sample_rate = sample_rate
		self.noises_source = zipfile.ZipFile('train_noisy.zip', 'r')
		self.noises = self.noises_source.namelist()

	def augmentate(self, file_to_mix):

		try:
			audio = AudioSegment.from_file(file_to_mix)
			noise = random.choice(self.noises)
			whirr = self.noises_source.extract(noise, self.folder_to_outputs)
			whirr = AudioSegment.from_file(whirr)
		except Exception as e:
			logging.error(traceback.format_exc())
		else:
			amplitude_difference = audio.dBFS-whirr.dBFS
			whirr = whirr.apply_gain(amplitude_difference)
			whirr = whirr.set_frame_rate(self.sample_rate)

			duration_audio = audio.duration_seconds
			duration_noise = whirr.duration_seconds
			difference = duration_audio/duration_noise
			if difference > 1:
				solutions = ['cycle', 'padding']
				solution = random.choice(solutions)
				if solution == 'cycle':
					while difference > 1:
						whirr = whirr + whirr
						difference = difference-1
				else:
					silence_mlls = (duration_audio-duration_noise)*1000
					silence = AudioSegment.silent(duration=silence_mlls)
					where_to_add_variants = ['left','right']
					where_to_add_variant = random.choice(where_to_add_variants)
					if where_to_add_variant == 'right':
						whirr = whirr + silence
					else:
						whirr = silence + whirr

			augmented_file = audio.overlay(whirr.apply_gain(-(self.noise_decibels)))
			head, tail = os.path.split(file_to_mix)
			augmented_file.export(f'{self.noise_decibels}/{self.folder_to_outputs}{tail}', format="wav")
			os.remove(f'{self.folder_to_outputs}{noise}')
