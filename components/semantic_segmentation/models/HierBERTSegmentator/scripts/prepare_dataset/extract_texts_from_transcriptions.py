# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
import json
import random
import pandas as pd
from tqdm import tqdm
from functools import reduce


# Пока здесь не учитывается обработка файлов реально большого размера

def extract_texts_from_transcriptions(input_dir, output_dir, silent_tqdm=False):
    # Get paths of all files
    file_paths = []
    for file_name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, file_name)
        file_paths.append(full_path)

    texts = []
    for file_path in tqdm(file_paths, disable=silent_tqdm):
        with open(file_path, encoding="utf-8") as file:
            texts.append(" ".join(file.readlines()))

    # Save output file
    with open(os.path.join(output_dir, "transcriptions_texts.json"), "w", encoding="utf-8") as output_file:
        json.dump(texts, output_file)


if __name__ == "__main__":
    # extract_texts_from_transcriptions("../../../Call_center_dialogs/Long_text_restored", "../data_raw/long_transcriptions")
    extract_texts_from_transcriptions("../../../Call_center_dialogs/Short_text_restored",
                                      "../data_raw/short_transcriptions")
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--input_dir", type=str, default="./",
    #                     help="Directory with text documents for dataset creation")
    # parser.add_argument("--output_dir", type=str, default="./",
    #                     help="Directory for prepared dateset")
    # parser.add_argument("--train_test_split", type=float, default=0.7,
    #                     help="Ratio of training and testing parts in dataset")
    #
    # args = parser.parse_args()
