# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from razdel import sentenize
import argparse

from models.HierBERTSegmentator.HierBERTSegmentator import HierBERTSegmentator


"""
Model initialization and inference function
"""


app = Flask(__name__)
model = None


def process_data(request_data):
    text, speaker_bounds = extract_text_and_speaker_bounds(request_data)
    # asr_data = request_data["diarization"]

    if "text" in request_data:
        text = request_data["text"]
    else:
        raise NotImplementedError
        # request_data["text"] = text

    words = text.split(" ")

    segments = model.predict(text)

    # Уточняем границы сегментов по границам спикеров
    segment_bounds = [0]
    for segment in segments:
        segment_bounds.append(segment_bounds[-1] + len(segment))
    segment_bounds = segment_bounds[1:-1]

    new_segment_bounds = []
    for segment_bound in segment_bounds:
        if len(new_segment_bounds) > 0 and new_segment_bounds[-1] >= segment_bound:
            continue
        for speaker_bound in speaker_bounds:
            if speaker_bound >= segment_bound:
                new_segment_bounds.append(speaker_bound)
                break

    segments = []
    if len(new_segment_bounds) == 0:
        segments.append(text)
    else:
        for i in range(len(new_segment_bounds)):
            if i == 0:
                segments.append(words[:new_segment_bounds[i]])
            else:
                segments.append(words[new_segment_bounds[i - 1]:new_segment_bounds[i]])

    request_data["segments"] = segments

    return request_data

    # Приводим данные к нужной структуре
    # result = {"result": [{"segment": []}]}
    # cur_segment = 0
    # speaker_segments_length = 0
    # for speaker_segment in asr_data:
    #     add_length = len([word["word"] for word in speaker_segment["words"]])
    #
    #     if speaker_segments_length + add_length < len(segments[cur_segment]):
    #         result["result"][-1]["segment"].append(speaker_segment)
    #         speaker_segments_length += add_length
    #     elif speaker_segments_length + add_length == len(segments[cur_segment]):
    #         result["result"][-1]["segment"].append(speaker_segment)
    #         result["result"][-1]["sentences"] = [x.text for x in sentenize(segments[cur_segment])]
    #         result["result"].append({"segment": []})
    #
    #         cur_segment += 1
    #         speaker_segments_length = 0
    # result["result"][-1]["sentences"] = [x.text for x in sentenize(segments[cur_segment])]

    # return result


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

    if (not isinstance(request_data, str)) and (not isinstance(request_data, dict)) and \
            (not isinstance(request_data, list)):
        err_msg = f'{type(request_data)} is unknown data type for emotion analyzer!'
        resp = jsonify({'message': err_msg})
        resp.status_code = 400
        return resp

    # try:
    #     result = process_data(request_data)
    #
    #     resp = jsonify(result)
    #     resp.status_code = 200
    # except Exception as e:
    #     err_msg = str(e)
    #     resp = jsonify({'message': err_msg})
    #     resp.status_code = 400

    result = process_data(request_data)

    resp = jsonify(result)
    resp.status_code = 200

    return resp


if __name__ == "__main__":
    # Инициаилизация модели
    # Параметры для моделей (название основной модели и входящих в нее моделей, разные настройки и т.д.)
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./models/HierBERTSegmentator/checkpoints/checkpoint-4000",
                        help="Directory with trained model")
    parser.add_argument("--sentence_encoder_path", type=str,
                        default="./models/HierBERTSegmentator/checkpoints/sentence_ru_cased_L-12_H-768_A-12_pt",
                        help="Directory with model for sentence encoding (BERT-based")
    parser.add_argument("--tokenizer_path", type=str, default="DeepPavlov/rubert-base-cased",
                        help="Directory with tokenizer for sentence encoding. If None, --sentence_encoder_path is used")
    parser.add_argument("--average_method", type=str, default="CLS",
                        help="Method for getting sentence embeddings from word embeddings")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum length of input sequence")
    parser.add_argument("--use_cuda", type=bool, default=False,
                        help="Use GPU to increase performance")

    args = parser.parse_args()

    model = HierBERTSegmentator(args.model_path, args.sentence_encoder_path, tokenizer_path=args.tokenizer_path,
                                average_method=args.average_method, max_seq_length=args.max_seq_length,
                                use_cuda=args.use_cuda)

    # Для запуска в контейнере
    app.run(host='0.0.0.0', port=8806)
