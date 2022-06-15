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

def extract_texts_from_rutweetcorp(input_dir, output_dir, silent_tqdm=False):
    # Get paths of all files
    file_paths = []
    for dir_name in os.listdir(input_dir):
        file_names = os.listdir(os.path.join(input_dir, dir_name))
        full_paths = [os.path.join(input_dir, dir_name, file_name) for file_name in file_names]
        file_paths.extend(full_paths)

    texts = []
    for file_path in tqdm(file_paths, disable=silent_tqdm):
        df = pd.read_csv(file_path)
        texts.extend(list(df["text"].values))

    # Save output file
    with open(os.path.join(output_dir, "rutweetcorp_texts.json"), "w", encoding="utf-8") as output_file:
        json.dump(texts, output_file)


if __name__ == "__main__":
    extract_texts_from_rutweetcorp("../RuTweetCorp", "../data_raw")
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
