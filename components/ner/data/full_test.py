import codecs
import json
import requests


if __name__ == "__main__":
    resp = requests.get('http://localhost:8977/ready')
    print(resp.status_code)

    source_data_name = 'input_example.json'
    true_result_name = 'output_example.json'
    with codecs.open(source_data_name, mode='r', encoding='utf-8') as fp:
        source_data = json.load(fp)
    with codecs.open(true_result_name, mode='r', encoding='utf-8') as fp:
        true_data = json.load(fp)

    resp = requests.post('http://localhost:8977/recognize', json=source_data)
    print(resp.status_code)
    recognized_data = resp.json()
    assert type(recognized_data) == type(true_data)
    assert 'result' in recognized_data
    assert len(recognized_data['result']) == len(true_data['result'])
    it = zip(recognized_data['result'], true_data['result'])
    for sample_idx, (recognized_, true_) in enumerate(it):
        assert 'segment' in recognized_
        assert 'text' in recognized_
        assert 'ners' in recognized_
        assert type(recognized_['segment']) == type(true_['segment'])
        assert len(recognized_['segment']) == len(true_['segment'])
        assert recognized_['text'] == true_['text']
        assert len(recognized_['ners']) > 0
        print('==========')
        print(f'Segment {sample_idx}')
        print('==========')
        print('')
        print('Text')
        print(recognized_['text'])
        print('')
        print('Recognized named entities')
        print(recognized_['ners'])
        print('')
