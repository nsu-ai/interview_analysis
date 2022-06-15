# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import argparse

from models.BertPunctRestorer.BertPunctRestorer import BertPunctRestorer


app = Flask(__name__)
model = None


def process_data(model, request_data):
    """
    Функция для преобразования входных данных (опционально) и прогона их через модель
    """

    text, speaker_bounds = extract_text_and_speaker_bounds(request_data)

    # Исправим ошибку с заменой некоторых символов на упрощенные
    # Так просто не работает, нужно исправлять в самой модели
    # words = text.split(" ")
    # for i in range(len(words)):
    #     for symb in ["й", "ё", "Й", "Ё"]:
    #         if symb in words[i]:
    #             symb_idx = words[i].find(symb)
    #             words[i] = words[i][:symb_idx] + symb + words[i][symb_idx + 1:]
    #     if words[i].isupper():
    #         words[i] = words.upper()
    #
    # punctuated = " ".join(words)
    # print(punctuated)

    text = model.predict(text)

    # Уточним границы предложений с помощью информации о разбиении по дикторам
    words = text.split(" ")
    for bound in speaker_bounds:
        if words[bound][-1].isalnum():
            words[bound] += "."
        elif words[bound][-1] in [".", "?", "!"]:
            pass
        else:
            words[bound] = words[bound][:-1] + "."
        if bound < len(words):
            words[bound + 1] = words[bound + 1].capitalize()

    # Приводим данные к нужной структуре
    result = request_data
    result["text"] = text

    return result


def extract_text_and_speaker_bounds(request_data):
    data = request_data["diarization"]
    words = []
    speaker_bounds = []
    for speaker_segment in data:
        if "words" in speaker_segment:
            for word in speaker_segment["words"]:
                words.append(word["word"])
            speaker_bounds.append(len(words) - 1)
        speaker_bounds = speaker_bounds[:-1]
    return " ".join(words), speaker_bounds


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/recognize', methods=['POST'])
def recognize():
    request_data = request.get_json()

    # Check data format

    try:
        result = process_data(model, request_data)

        resp = jsonify(result)
        resp.status_code = 200
    except Exception as e:
        err_msg = str(e)
        resp = jsonify({'message': err_msg})
        resp.status_code = 400

    return resp


if __name__ == "__main__":
    # Инициаилизация модели
    # Параметры для моделей (название основной модели и входящих в нее моделей, разные настройки и т.д.)
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./models/BertPunctRestorer/checkpoints/checkpoint-2000",
                        help="Directory with trained model")
    parser.add_argument("--labels_path", type=str, default="./models/BertPunctRestorer/checkpoints/checkpoint-2000/labels.json",
                        help="Text file with labels")

    args = parser.parse_args()

    model = BertPunctRestorer(model_path=args.model_path, labels_path=args.labels_path)

    app.run(host='0.0.0.0', port=8803)
