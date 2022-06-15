# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import json
import random
import csv
import gc
from functools import reduce
import pandas as pd
from tqdm import tqdm

# Добавить функцию для фильтрации предложений, где мало знаков препинания (с учетом редких знаков, их не нужно пропускать)
# Нужно проверить эту гипотезу - связь между началом и концом предложений может влиять на результат

# TODO:
#  1. оптимизировать это дело
#  2. добавить аргументы
#  3. описания
#  4. тайпинги
#  6. Добавить логгинг
#  7. Подумать об исправлении дисбаланса классов
#  8. Перевести метки из собственного JSON в JSON с конфигом и добавить метки по умолчанию


# Эти метки связаны
with open('../labels.json') as json_file:
    LABELS = json.load(json_file)
# LABELS = ["OO", "UO", "O.", "O,", "O!", "O?", "O:", "O;", "O-", "U.", "U,", "U!", "U?", "U:", "U;", "U-"]
# Возможно, стоит добавить токен для явного начала или конца предложения


def clear_text(text, min_line_length=10):
    # Clean all extra new lines, double whitespaces and "void" lines
    text = text.replace("\n", " ")
    while "  " in text:
        text = re.sub(r"(\s)+", " ", text)

    # Clean all "too short" paragraphs
    # text = "\n".join([line for line in text.split("\n") if len(line) >= min_line_length])

    # Clean all sentences not ended with .?! - под вопросом

    return text


def split_into_samples(labeled_tokens, num_tokens=500, offset=250):
    """
    Given a large set of tokens, determines splits of
    500 token sized observations, with an offset(sliding window) of 250 tokens.
    It is important that first token is capitalized (rough approximation of sentence start)
    and we fed as many tokens as possible.
    In a real use-case we will not know where splits are so we'll just feed all tokens till limit.
    """
    start = -1
    loop_end = -1
    splits = []

    for ix, labeled_token in enumerate(labeled_tokens):
        if ix == loop_end:
            start = -1
        if labeled_token[1][0] == "U" and start == -1:
            start = ix
            end = ix + num_tokens
            splits.append((start, end))
            loop_end = start + offset

    return splits


def add_labels(text):
    """
    Create labels for Punctuation Restoration task for each token ("tokens" are splitted by whitespaces).

    Пока работает только со знаками на конце слов
    """
    labeled_tokens = []
    i = 0
    for token in tqdm(text.split(), position=1, leave=False, desc="add_labels"):
        normalized_token = re.sub(r"[\W_]", "", token).lower()

        if i % 100000 == 0:
            print(sys.getsizeof(labeled_tokens) / 1024 / 1024)
            import gc

            gc.collect()
        i += 1

        if not normalized_token:
            continue

        label = ""

        if token[0].isupper():
            label += "U"
        else:
            label += "O"

        if not token[-1].isalnum():
            label += token[-1]
        else:
            label += "O"

        if label not in LABELS:
            label = label[0] + "O"

        labeled_tokens.append([normalized_token, label])

    return labeled_tokens


def get_label_stats(dataset):
    """
    Generates frequency of different labels in the dataset.
    """
    res = dict([(label, 0) for label in LABELS])
    for sample in dataset:
        for token, label in sample:
            res[label] += 1

    # Works in Python 3.7+
    res = dict(reversed(sorted(res.items(), key=lambda item: item[1])))

    return res


def prepare_dataset(input_dir, output_dir, seed=None):
    """
    Ответственность за разбиение на трэйн/тест несет скрипт по подготовке сырых текстов

    Parameters
    ----------
    input_dir
    output_dir

    Returns
    -------

    """

    for file_name in tqdm(os.listdir(input_dir), position=0, leave=True):
        with open(os.path.join(input_dir, file_name), encoding="utf-8") as file:
            text = "\n".join(file.readlines())
            text = clear_text(text)#[:10000]

            print(sys.getsizeof(text) / 1024 / 1024)
            # 718 / 2832

            print("Start text labeling...")
            text = add_labels(text)

            gc.collect()

            print(sys.getsizeof(text) / 1024 / 1024)
            # 199 / 819

            splits = split_into_samples(text)

            print(sys.getsizeof(splits) / 1024 / 1024)
            # 0.78 / 3.23

            gc.collect()

            samples = [text[s[0]:s[1]] for s in splits]

            print(sys.getsizeof(samples) / 1024 / 1024)
            # 0.78 / 3.23

            if seed is not None:
                random.Random(seed).shuffle(samples)

            label_stat = get_label_stats(samples)
            num_examples = reduce(lambda x, y: x + y, label_stat.values())

            print("Label stat:\n")
            for label, count in label_stat.items():
                print(f"{label} : {count} ({count / num_examples:.6f})")

            gc.collect()

            output_file_name = os.path.join(output_dir, file_name.split(".")[0] + ".csv")
            with open(output_file_name, "w", encoding="utf-8", newline="") as f_out:
                # conll2003 tokens are space separated
                writer = csv.writer(f_out, delimiter=' ')
                # writer.writerow(["tokens", "punct_restoration_tags"])
                for sample in samples:
                    writer.writerows(sample)
                    writer.writerow("")
           

if __name__ == "__main__":
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
    #
    # for file in os.listdir(args.input_dir):
    #     create_rpunct_dataset(file, args.output_dir)
    #     create_training_samples(file, args.output_dir)

    prepare_dataset("../data_raw", "../data")

    # from datasets import ClassLabel, load_dataset
    #
    # data_files = {}
    # data_files["train"] = "../data/train.csv"
    # data_files["validation"] = "../data/test.csv"
    #
    # extension = "../data/train.csv".split(".")[-1]
    #
    # raw_datasets = load_dataset("punct_restoration_dataset.py", data_files=data_files)
    # # raw_datasets = load_dataset("conll2003")
    #
    # column_names = raw_datasets["train"].column_names
    # features = raw_datasets["train"].features
    #
    # text_column_name = column_names[0]
    # label_column_name = column_names[1]
    #
    # # if isinstance(features[label_column_name].feature, ClassLabel):
    # #     pass