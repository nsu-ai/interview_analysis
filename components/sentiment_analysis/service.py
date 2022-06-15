# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import argparse

from models.RuBERTSentiment.RuBERTSentiment import RuBERTSentiment


app = Flask(__name__)
model = None


def process_data(model, request_data):
    """
    Функция для преобразования входных данных (опционально) и прогона их через модель
    """

    if "summaries" in request_data:
        request_data["sentiments"] = [model.predict(sum_text) for sum_text in request_data["summaries"]]
    elif "segments" in request_data:
        request_data["sentiments"] = [model.predict(segm_text) for segm_text in request_data["segments"]]
    elif "text" in request_data:
        request_data["sentiments"] = [model.predict(request_data["text"])]
    else:
        raise NotImplementedError

    return request_data


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

    # parser.add_argument("--model_path", type=str, default="blanchefort/rubert-base-cased-sentiment",
    #                     help="Directory with trained model")
    parser.add_argument("--model_path", type=str, default="./models/RuBERTSentiment/checkpoints/blanchefort_rubert-base-cased-sentiment",
                        help="Directory with trained model")

    args = parser.parse_args()

    model = RuBERTSentiment(model_path=args.model_path)

    # Для запуска в контейнере
    app.run(host='0.0.0.0', port=8805)
