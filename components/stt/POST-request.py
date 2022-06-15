import sys
import requests


def main():
	requests.get('http://localhost:8802/ready')  # проверяем готовность системы
	if len(sys.argv) > 1:
		sound_name = sys.argv[1]
	else:
		sound_name = 'test.wav'
	files = {'wav': (sound_name, open(sound_name, 'rb'), 'audio/wave')}
	resp = requests.post('http://localhost:8802/recognize', files=files)
	print(resp.json())


if __name__ == '__main__':
	main()
