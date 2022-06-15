# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
import json
import random
import csv
from scipy.stats import norm
import numpy as np
import pandas as pd
from tqdm import tqdm
from razdel import sentenize
from sklearn.model_selection import train_test_split
from functools import reduce


def clear_text(text):
    # Clean all extra new lines, double whitespaces and "void" lines
    text = text.replace("\n", " ")
    while "  " in text:
        text = re.sub(r"(\s)+", " ", text)

    return text


def gen_samples_ids(text_to_sent_num,
                    sent_num_to_texts,
                    with_repeats=False,
                    max_samples=-1,
                    min_topics_num=1,
                    max_topics_num=10,
                    min_sentences_num=2,
                    max_sentences_num=10):
    all_topics_nums = np.array(range(min_topics_num, max_topics_num))
    all_sentences_nums = np.array(range(min_sentences_num, max_sentences_num))

    # Calculate weight for topics and sentences
    topic_scale = 1.0
    while norm.pdf(all_topics_nums, loc=all_topics_nums.mean(), scale=topic_scale).sum() > 0.9:
        topic_scale += 0.1
    sent_scale = 1.0
    while norm.pdf(all_sentences_nums, loc=all_sentences_nums.mean(), scale=sent_scale).sum() > 0.9:
        sent_scale += 0.1

    topic_weights = norm.pdf(all_topics_nums, loc=all_topics_nums.mean(), scale=topic_scale)
    sent_weights = norm.pdf(all_sentences_nums, loc=all_sentences_nums.mean(), scale=sent_scale)

    samples = []
    while True:
        if len(samples) == max_samples:
            break

        topic_num = random.choices(population=all_topics_nums, weights=topic_weights, k=1)[0]
        sentences_nums = random.choices(population=all_sentences_nums, weights=sent_weights, k=topic_num)

        sample = []
        not_enough_sent = False
        for i in range(topic_num):
            # Choose random group of texts with number of sentences >= sentences_nums[i]
            groups = [k for k in sent_num_to_texts.keys() if k >= sentences_nums[i]]
            if not groups:
                not_enough_sent = True
                break

            group = random.sample(groups, 1)[0]

            # Choose random text
            text = random.sample(sent_num_to_texts[group], 1)[0]

            if with_repeats:
                # Get random index of start sentence and don`t remove indices of chosen sentences from pool
                sample.append(
                    {"text_id": text,
                     "start_sent_id": random.randint(0, text_to_sent_num[text]["sent_num"] - sentences_nums[i]),
                     "sent_num": sentences_nums[i]
                     })
            else:
                # Get indices of sentences and remove this indices from pool
                sample.append(
                    {"text_id": text,
                     "start_sent_id": text_to_sent_num[text]["sent_id"],
                     "sent_num": sentences_nums[i]
                     })

                text_to_sent_num[text]["sent_id"] += sentences_nums[i]
                text_to_sent_num[text]["sent_num"] -= sentences_nums[i]

                sent_num_to_texts[group].remove(text)
                if not sent_num_to_texts[group]:
                    del sent_num_to_texts[group]

                if text_to_sent_num[text]["sent_num"] not in sent_num_to_texts:
                    sent_num_to_texts[text_to_sent_num[text]["sent_num"]] = []
                sent_num_to_texts[text_to_sent_num[text]["sent_num"]].append(text)

        if not_enough_sent:
            print(f"Not enough sentences for generation. Current number of samples: {len(samples)}")
            break

        samples.append(sample)

    return samples


def prepare_segmentation_dataset(input_file,
                                 output_dir,
                                 max_samples=-1,
                                 min_topics_num=1,
                                 max_topics_num=10,
                                 min_sentences_num=2,
                                 max_sentences_num=12,
                                 with_repeats=False,
                                 test_size=None,
                                 train_size=None,
                                 random_state=42,
                                 silent_tqdm=False):
    """


    """
    with open(input_file, encoding="utf-8") as file:
        texts = json.load(file)

    # Split texts in sentences
    sentenized_texts = []
    for text in tqdm(texts, disable=silent_tqdm):
        sentenized_texts.append(list(sentenize(clear_text(text))))
    del texts

    # Fow each text we need information about remaining sentences and index of last used sentence
    text_to_sent_num = dict([(i, {"sent_id": 0, "sent_num": len(text)}) for i, text in enumerate(sentenized_texts)])
    sent_num_to_texts = dict()
    for k, v in text_to_sent_num.items():
        if v["sent_num"] not in sent_num_to_texts:
            sent_num_to_texts[v["sent_num"]] = []
        sent_num_to_texts[v["sent_num"]].append(k)
    sent_num_to_texts = dict(sorted(sent_num_to_texts.items(), key=lambda item: item[0]))

    # Print number of sentences of each length and mean possible number of fully unique samples
    print("\nNumber of sentences of each length:")
    all_sentence_num = 0
    for k, v in sent_num_to_texts.items():
        print(k, len(v))
        all_sentence_num += k * len(v)
    mean_sample_len = (min_topics_num + (max_topics_num - min_topics_num) / 2) * \
                              (min_sentences_num + (max_sentences_num - min_sentences_num) / 2)
    print(f"\nMean possible number of samples: {int(all_sentence_num / mean_sample_len)}\n")

    # Generate samples as ids
    print("Samples ids are generated, please wait...")
    samples_ids = gen_samples_ids(text_to_sent_num, sent_num_to_texts, with_repeats=with_repeats,
                                  max_samples=max_samples)

    # Split into train and test samples
    train, test = train_test_split(samples_ids, test_size=test_size, train_size=train_size,
                                   shuffle=False, random_state=random_state)

    # Generate and save full samples - with labels and sentences instead of ids
    for name, samples in [("train", train), ("test", test)]:
        with open(os.path.join(output_dir, f"{name}_{len(samples)}.csv"), "w", newline='', encoding="utf-8") as output_file:
            writer = csv.writer(output_file)

            first_row = True
            for sample in tqdm(samples, desc=f"Prepare {name}", disable=silent_tqdm):
                labeled_sample = {"sentences": [], "labels": []}
                for i in sample:
                    sentences = [s.text for s in
                                 sentenized_texts[i["text_id"]][i["start_sent_id"]:i["start_sent_id"] + i["sent_num"]]]

                    labeled_sample["sentences"].extend(sentences)

                    labels = [0] * i["sent_num"]
                    if labeled_sample["labels"]:
                        labels[0] = 1
                    labeled_sample["labels"].extend(labels)

                if first_row:
                    first_row = False
                else:
                    writer.writerow("")
                writer.writerows(zip(labeled_sample["sentences"], labeled_sample["labels"]))


if __name__ == "__main__":
    prepare_segmentation_dataset("../data_raw/long_transcriptions/transcriptions_texts.json",
                                 "../data/audio_sentences",
                                 max_samples=10000,
                                 min_topics_num=1,
                                 max_topics_num=10,
                                 min_sentences_num=3,
                                 max_sentences_num=12,
                                 with_repeats=True,
                                 test_size=0.3,
                                 train_size=0.7,
                                 random_state=42,
                                 silent_tqdm=False,
                                 )

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--input_file", type=str, default="./",
    #                     help="Json file with text documents for dataset creation")
    # parser.add_argument("--output_dir", type=str, default="./",
    #                     help="Directory for prepared dateset")
    # parser.add_argument("--max_samples", type=int, default=-1,
    #                     help="Maximum number of generated samples. If '-1', then there are no restrictions")
    # parser.add_argument("--min_topics_num", type=int, default=1,
    #                     help="Minimum number of topics in generated samples")
    # parser.add_argument("--max_topics_num", type=int, default=10,
    #                     help="Maximum number of topics in generated samples")
    # parser.add_argument("--min_sentences_num", type=int, default=2,
    #                     help="Minimum number of sentences in each topic in generated samples")
    # parser.add_argument("--max_sentences_num", type=int, default=12,
    #                     help="Maximum number of sentences in each topic in generated samples")
    # parser.add_argument("--with_repeats", type=bool, default=False,
    #                     help="Using sentence more than once for sample generation")
    # parser.add_argument("--test_size", default=None,
    #                     help="Scikit-learn like test_size for train_test_split")
    # parser.add_argument("--train_size", default=None,
    #                     help="Scikit-learn like train_size for train_test_split")
    # parser.add_argument("--random_state", default=None,
    #                     help="Random state for train_test_split")
    # parser.add_argument("--silent_tqdm", type=bool, default=False,
    #                     help="Switch to show progress bar")
    #
    # args = parser.parse_args()
    #
    # prepare_segmentation_dataset(args.input_file,
    #                              args.output_dir,
    #                              max_samples=args.max_samples,
    #                              min_topics_num=args.min_topics_num,
    #                              max_topics_num=args.max_topics_num,
    #                              min_sentences_num=args.min_sentences_num,
    #                              max_sentences_num=args.max_sentences_num,
    #                              test_size=args.test_size,
    #                              train_size=args.train_size,
    #                              random_state=args.random_state,
    #                              silent_tqdm=args.silent_tqdm,
    #                              )


"""
TODO

1. Брать предложения с повторениями 
"""