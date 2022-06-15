# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import argparse

from models.MBartRuSumGazeta.MBartRuSumGazeta import MBartRuSumGazeta


app = Flask(__name__)
model = None


def process_data(model, request_data):
    """
    Функция для преобразования входных данных (опционально) и прогона их через модель
    """
    if "segments" in request_data:
        request_data["summaries"] = [model.predict(segm_text) for segm_text in request_data["segments"]]
    elif "text" in request_data:
        request_data["summaries"] = [model.predict(request_data["text"])]
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

    # parser.add_argument("--model_path", type=str, default="IlyaGusev/mbart_ru_sum_gazeta",
    #                     help="Directory with trained model or model name in public hub")
    parser.add_argument("--model_path", type=str, default="./models/MBartRuSumGazeta/checkpoints/IlyaGusev_mbart_ru_sum_gazeta",
                        help="Directory with trained model or model name in public hub")

    args = parser.parse_args()

    model = MBartRuSumGazeta(model_path=args.model_path)

    # Для локального тестирования
    # app.run(host='localhost', port=8807)
    # Для запуска в контейнере
    app.run(host='0.0.0.0', port=8807)
