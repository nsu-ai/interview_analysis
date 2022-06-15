# -*- coding: utf-8 -*-
import copy
import logging
import os
import shutil
from typing import List, Tuple, Union
import zipfile

from flask import Flask, request, jsonify
import requests
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from urllib.parse import urlencode

from scripts.data_processing.tokenization import tokenize_text, sentenize_text
from scripts.data_processing.postprocessing import decode_entity
from scripts.neural_network.ner import load_ner


ner_logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
app = Flask(__name__)
ner_model = None
ner_tokenizer = None
max_sent_len = None
ne_list = None


def download_ner() -> bool:
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://yadi.sk/d/7CQPhR2SAu6mxw'
    final_url = base_url + urlencode(dict(public_key=public_key))
    pk_request = requests.get(final_url)
    direct_link = pk_request.json().get('href')
    response = requests.get(direct_link, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    ner_logger.info(f'Total size of NER is {total_size_in_bytes} bytes.')
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    zip_archive_name = os.path.join(model_path, 'dp_rubert_from_siamese.zip')
    with open(zip_archive_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if (total_size_in_bytes != 0) and (progress_bar.n != total_size_in_bytes):
        return False
    with zipfile.ZipFile(zip_archive_name) as archive:
        archive.extractall(model_path)
    os.remove(zip_archive_name)
    return True


def recognize_single_text(cur_text: str) -> List[Tuple[int, int, str]]:
    global ner_model, ner_tokenizer, max_sent_len, ne_list
    recognized_entities = []
    if len(cur_text.strip()) > 0:
        for sent_start, sent_end in sentenize_text(cur_text):
            words, subtokens, subtoken_bounds = tokenize_text(
                s=cur_text[sent_start:sent_end],
                tokenizer=ner_tokenizer
            )
            while (len(subtokens) % max_sent_len) != 0:
                subtokens.append(ner_tokenizer.pad_token)
                subtoken_bounds.append(None)
            x = []
            start_pos = 0
            for _ in range(len(subtokens) // max_sent_len):
                end_pos = start_pos + max_sent_len
                subtoken_indices = ner_tokenizer.convert_tokens_to_ids(
                    subtokens[start_pos:end_pos]
                )
                x.append(
                    np.array(
                        subtoken_indices,
                        dtype=np.int32
                    ).reshape((1, max_sent_len))
                )
                start_pos = end_pos
            predicted = ner_model.predict(np.vstack(x), batch_size=1)
            if len(predicted) != len(ne_list):
                err_msg = f'Number of neural network heads does not ' \
                          f'correspond to number of named entities! ' \
                          f'{len(predicted)} != {len(ne_list)}'
                raise ValueError(err_msg)
            del x
            probability_matrices = [
                np.vstack([
                    cur[sample_idx]
                    for sample_idx in range(len(subtokens) // max_sent_len)
                ])
                for cur in predicted
            ]
            del predicted
            for ne_idx in range(len(ne_list)):
                entity_bounds = decode_entity(
                    softmax(probability_matrices[ne_idx], axis=1),
                    words
                )
                if len(entity_bounds) > 0:
                    for start_subtoken, end_subtoken in entity_bounds:
                        entity_start = subtoken_bounds[start_subtoken][0]
                        entity_end = subtoken_bounds[end_subtoken - 1][1]
                        recognized_entities.append((
                            sent_start + entity_start,
                            sent_start + entity_end,
                            ne_list[ne_idx]
                        ))
                del entity_bounds
            del words, subtokens, subtoken_bounds
    return recognized_entities


def process_data(request_data):
    """
    Функция для преобразования входных данных (опционально) и прогона их через модель
    """

    if (not isinstance(request_data, str)) and \
            (not isinstance(request_data, dict)):
        err_msg = f'{type(request_data)} is unknown data type for ' \
                  f'the named entity recognizer!'
        raise ValueError(err_msg)
    if isinstance(request_data, str):
        result = recognize_single_text(request_data)
    else:
        named_entities = []
        if len(request_data) == 0:
            err_msg = 'The input data are empty!'
            raise ValueError(err_msg)
        if 'result' not in request_data:
            err_msg = 'The input data are wrong! ' \
                      'The expected key `result` is not found!'
            raise ValueError(err_msg)
        if not isinstance(request_data['result'], list):
            err_msg = f'The input data are wrong! Expected {type([1, 2])}, ' \
                      f'got {type(request_data["result"])}!'
            raise ValueError(err_msg)
        if len(request_data['result']) > 0:
            for segment_idx, cur_segment in enumerate(request_data['result']):
                if not isinstance(cur_segment, dict):
                    err_msg = f'The input segment {segment_idx} is wrong! ' \
                              f'Expected {type({"a": 1, "b": 2})}, ' \
                              f'got {type(cur_segment)}!'
                    raise ValueError(err_msg)
                if 'text' not in cur_segment:
                    err_msg = f'The input segment {segment_idx} is wrong! ' \
                              f'The expected key `text` is not found!'
                    raise ValueError(err_msg)
                source_text = cur_segment['text']
                named_entities.append(recognize_single_text(source_text))
        result = copy.deepcopy(request_data)
        for segment_idx in range(len(result['result'])):
            result['result'][segment_idx]['ners'] = copy.deepcopy(
                named_entities[segment_idx]
            )
        del named_entities

    return result


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/recognize', methods=['POST'])
def recognize():
    request_data = request.get_json()

    try:
        result = process_data(request_data)

        resp = jsonify(result)
        resp.status_code = 200
    except Exception as e:
        err_msg = str(e)
        resp = jsonify({'message': err_msg})
        resp.status_code = 400
        ner_logger.error(err_msg)

    return resp


if __name__ == "__main__":
    # Инициализация модели
    # Параметры для моделей (название основной модели и входящих в нее моделей, разные настройки и т.д.)
    # Нужно обрабатывать с помощью argparser
    model_path = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.isdir(model_path):
        raise ValueError(f'The directory "{model_path}" does not exist!')
    trained_ner_path = os.path.join(model_path, 'dp_rubert_from_siamese')
    if not os.path.isdir(trained_ner_path):
        ner_exists = False
    else:
        if not os.path.isfile(os.path.join(trained_ner_path, 'ner.h5')):
            ner_exists = False
        elif not os.path.isfile(os.path.join(trained_ner_path, 'ner.json')):
            ner_exists = False
        else:
            ner_exists = True
    if not ner_exists:
        if os.path.isdir(trained_ner_path):
            shutil.rmtree(trained_ner_path, ignore_errors=True)
        if not download_ner():
            raise ValueError('The NER cannot be downloaded from Yandex Disk!')
    if not os.path.isdir(trained_ner_path):
        raise ValueError(f'The directory "{trained_ner_path}" does not exist!')
    ner_model, ner_tokenizer, max_sent_len, ne_list = load_ner(trained_ner_path)

    # Для локального тестирования
    # app.run(host='localhost', port=8977)
    # Для запуска в контейнере
    app.run(host='0.0.0.0', port=8977)
