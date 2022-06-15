# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse
import json
import random
import csv
import gc
import pandas as pd
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
#  9. Сделать отдельную версию для очень больших файлов и маленькой оперативки


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


def split_into_samples(df, num_tokens=300, offset=250):
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

    for ix, row in tqdm(df.iterrows(), total=len(df), position=1, leave=False, desc="split_into_samples"):
        if ix == loop_end:
            start = -1
        if row["label"][0] == "U" and start == -1:
            start = ix
            end = ix + num_tokens
            splits.append((start, end))
            loop_end = start + offset

    return splits


def add_labels(df, allowed_labels):
    """
    Create labels for Punctuation Restoration task for each token ("tokens" are splitted by whitespaces).

    Пока работает только со знаками на конце слов
    """
    i = 0

    for ix, row in tqdm(df.iterrows(), total=len(df), position=1, leave=False, desc="add_labels"):
        token = row["token"]
        normalized_token = re.sub(r"[\W_]", "", token).lower()

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

        if label not in allowed_labels:
            label = label[0] + "O"

        df.at[ix, "label"] = label
        df.at[ix, "token"] = normalized_token

    return df


def get_label_stats(dataset, labels):
    """
    Generates frequency of different labels in the dataset.
    """
    res = dict([(label, 0) for label in labels])
    for sample in dataset:
        for _, row in sample.iterrows():
            res[row["label"]] += 1

    # Works in Python 3.7+
    res = dict(reversed(sorted(res.items(), key=lambda item: item[1])))

    return res


def prepare_dataset(input_dir, output_dir, labels_path, seed=None):
    """
    Ответственность за разбиение на трэйн/тест несет скрипт по подготовке сырых текстов

    Parameters
    ----------
    input_dir
    output_dir

    Returns
    -------

    """
    with open(labels_path) as labels_file:
        labels = json.load(labels_file)

    for file_name in tqdm(os.listdir(input_dir), position=0, leave=True):
        print(file_name)
        with open(os.path.join(input_dir, file_name), encoding="utf-8") as file:
            text = "\n".join(file.readlines())
            text = clear_text(text)

            df = pd.DataFrame(text.split(), columns=["token"])
            df["label"] = pd.NA

            print("Start text labeling...")
            df = add_labels(df, labels)
            df = df.dropna().reset_index(drop=True)

            splits = split_into_samples(df)

            print("Preparing samples...")
            samples = [df.loc[s[0]:s[1]] for s in splits]

            if seed is not None:
                random.Random(seed).shuffle(samples)

            print("Preparing label stat...")
            label_stat = get_label_stats(samples, labels)
            num_examples = reduce(lambda x, y: x + y, label_stat.values())

            print("Label stat:\n")
            for label, count in label_stat.items():
                print(f"{label} : {count} ({count / num_examples:.6f})")

            output_file_name = os.path.join(output_dir, file_name.split(".")[0] + ".csv")
            with open(output_file_name, "w", encoding="utf-8", newline="") as f_out:
                # conll2003 tokens are space separated
                writer = csv.writer(f_out, delimiter=' ')
                for sample in samples:
                    writer.writerows(sample.values)
                    writer.writerow("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="./",
                        help="Directory with text documents for dataset creation")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory for prepared dateset")
    parser.add_argument("--labels_path", type=str, default="./",
                        help="Path to file with allowed labels")

    args = parser.parse_args()

    prepare_dataset(args.input_dir, args.output_dir, args.labels_path)
