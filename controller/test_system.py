import json
from flask import jsonify
import codecs
import requests
import sys
import time
import re
import os

def main():
    adress = ''
    sound_name = ''
    default_name = 'test1.wav'
    len_args = len(sys.argv)
    if len_args > 1:
        if len_args == 3:
            sound_name = sys.argv[1]
            adress = sys.argv[2]
        elif len_args == 2:
            smth = sys.argv[1]
            opt = re.search(r'\w\w\w', smth)
            if opt != None:
                sound_name = smth
            else:
                adress = smth
    status = True
    while status:
        try:
            if adress != '':
                resp = requests.get('http://'+adress+':8800/ready')
            else:
                resp = requests.get('http://localhost:8800/ready')
            if resp.status_code != 200:
                time.sleep(10)
            else:
                status = False
        except:
            time.sleep(10)
    #files = {'audio': (sound_name, open(sound_name, 'rb'), 'audio/wave')}
    #  options in string format: diar/stt/seg/psy
    #options_ = 'diar/stt'
    files = None
    if not status:
        options_ = None
        if sound_name != '':
            pass
        else:
            sound_name = default_name
        if options_ != None:
            files = {'audio': (sound_name, open(sound_name, 'rb'), 'audio/wave'), 'options': jsonify({'options': options_})}
        else:
            files = {'audio': (sound_name, open(sound_name, 'rb'), 'audio/wave')}
    #json=json.dumps(options_)
        if adress != '':
            resp = requests.post('http://'+adress+':8800/recognize', files=files)
        else:
            resp = requests.post('http://localhost:8800/recognize', files=files)
        print('Request time = ' + str(resp.elapsed))
        result_dict = resp.json()
        save_file = os.path.splitext(sound_name)[0] + '_result.json'
        with codecs.open(save_file, mode='w', encoding='utf-8', errors='ignore') as out_f:
            json.dump(result_dict, fp=out_f, ensure_ascii=False)

        for key in result_dict:
            print('==========')
            print(key)
            print('==========')
            print(result_dict[key])
        #     if key1 == 'message':
        #         print(f'    {result_dict[key1]}')
        #     else:
        #         for key2 in result_dict[key1]:
        #             print(f'  {key2}:')
        #             for item_idx, cur_item in enumerate(result_dict[key1][key2]):
        #                 keys_of_item = set(cur_item.keys())
        #                 if 'speaker' in keys_of_item:
        #                     print(f'    segment {item_idx}, speaker {cur_item["speaker"]}')
        #                     keys_of_item -= {'speaker'}
        #                 else:
        #                     print(f'    {item_idx}')
        #                 if ('start_time' in keys_of_item) and ('end_time' in keys_of_item):
        #                     print('      start_time = {0:.3f}, end_time = {1:.3f}'.format(cur_item['start_time'], cur_item['end_time']))
        #                     keys_of_item -= {'start_time', 'end_time'}
        #                 if 'words' in keys_of_item:
        #                     printed_text = ''
        #                     for cur_word in cur_item['words']:
        #                         if isinstance(cur_word, dict):
        #                             if 'word' in cur_word:
        #                                 printed_text += f' {cur_word["word"]}'
        #                     printed_text = printed_text.strip()
        #                 #     if len(printed_text) > 0:
        #                 #         print(f'      text  = {printed_text}')
        #                 #     print('      words = {0}'.format(cur_item['words']))
        #                 #     keys_of_item -= {'words'}
        #                 # for key3 in sorted(list(keys_of_item)):
        #                 #     print(f'        {key3} = {cur_item[key3]}')


if __name__ == '__main__':
    main()