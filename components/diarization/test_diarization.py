import sys
import requests
import base64
import struct

# Это файл-пример теста модуля диаризации.
# Здесь полученные в результате сепарации аудио сохраняются в текущую директорию. И только для двух дикторов.
# Чтобы сохранять для N дикторов, необходимо небольшое изменение - добавление цикла.

def main():
    requests.get('http://localhost:8801/ready')  # проверяем готовность системы
    if len(sys.argv) > 1:
        sound_name = sys.argv[1]
    else:
        sound_name = 'test1.wav'
    files = {'wav': (sound_name, open(sound_name, 'rb'), 'audio/wave')}
    resp = requests.post('http://localhost:8801/recognize', files=files)
    result_json = resp.json()
    print(result_json["diarization"])
    bytes_json_zero = result_json["dictor_0_audio"].encode("ascii")
    encoded_zero = base64.b64decode(bytes_json_zero)
    with open("dictor_zero.wav", "wb") as outf:
        outf.write(encoded_zero)
    bytes_json_first = result_json["dictor_1_audio"].encode("ascii")
    encoded_first = base64.b64decode(bytes_json_first)
    with open("dictor_first.wav", "wb") as outz:
        outz.write(encoded_first)


if __name__ == '__main__':
	main()
