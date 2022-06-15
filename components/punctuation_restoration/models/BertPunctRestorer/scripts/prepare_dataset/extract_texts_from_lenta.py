# -*- coding: utf-8 -*-

import os
import re
import argparse
import json
import random
import pandas as pd
from tqdm import tqdm
from functools import reduce


# Пока здесь не учитывается обработка файлов реально большого размера

# TODO: change train-test-split arguments style like a sklearn train_test_split arguments style

def extract_texts_from_lenta(input_dir, output_dir, test_size=None, train_size=None, val_size=None, silent_tqdm=False):
    """
    Lenta dataset has file structure:
    lenta:
    - 1914-lenta:
    -- 1914-09-lenta.ru.csv

    Parameters
    ----------
    input_dir
    output_dir
    test_size
    train_size
    val_size
    silent_tqdm

    Returns
    -------

    """

    # Get paths of all files
    file_paths = []
    for dir_name in os.listdir(input_dir):
        file_names = os.listdir(os.path.join(input_dir, dir_name))
        full_paths = [os.path.join(input_dir, dir_name, file_name) for file_name in file_names]
        file_paths.extend(full_paths)

    # Combine all texts into one
    text = []
    for file_path in tqdm(file_paths, disable=silent_tqdm):
        df = pd.read_csv(file_path)
        combined_text = "\n".join(df["text"].values)
        text.append(combined_text)
    text = "\n".join(text)

    num_symbols = len(text)
    text = text.split("\n")
    num_lines = len(text)

    # Process data splitting arguments
    test_size = int(0.2 * num_lines) if test_size is None else \
        int(test_size * num_lines) if test_size <= 1.0 else test_size

    train_size = num_lines - test_size if train_size is None else \
        int(train_size * num_lines) if train_size <= 1.0 else train_size

    val_size = 0 if val_size is None else \
        int(val_size * num_lines) if val_size <= 1.0 else val_size

    assert test_size + train_size + val_size <= num_lines, "Error in data splitting"

    train_part = "\n".join(text[:train_size])
    test_part = "\n".join(text[train_size:train_size + test_size])
    val_part = "\n".join(text[train_size + test_size:])

    print(f"Data splitting:\n"
          f"Train:\t{len(train_part)} symbols ({(len(train_part) / num_symbols):.2f})\n"
          f"Test:\t{len(test_part)} symbols ({(len(test_part) / num_symbols):.2f})\n"
          f"Val:\t{len(val_part)} symbols ({(len(val_part) / num_symbols):.2f})\n")

    # Save output files
    for part, file_name in zip([train_part, test_part, val_part],
                               ["train.txt", "test.txt", "val.txt"]):
        if len(part) > 0:
            with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as output_file:
                output_file.write(part)


if __name__ == "__main__":
    extract_texts_from_lenta("../lenta", "../data_raw")
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
