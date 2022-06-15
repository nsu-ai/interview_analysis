# -*- coding: utf-8 -*-

import logging
import json
import argparse
from transformers import MBartTokenizer, MBartForConditionalGeneration
import torch


class MBartRuSumGazeta:
    def __init__(self, model_path="IlyaGusev/mbart_ru_sum_gazeta"):
        self.tokenizer = MBartTokenizer.from_pretrained(model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(model_path)

    def predict(self, text: str):
        """

        """

        input_ids = self.tokenizer(
            [text],
            max_length=600,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]

        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="IlyaGusev/mbart_ru_sum_gazeta",
                        help="Directory with trained model or model name in public hub")
    parser.add_argument("--text", type=str, default="",
                        help="Input text for punctuation restoration")

    args = parser.parse_args()

    model = MBartRuSumGazeta(model_path=args.model_path)

    summary = model.predict(args.text)

    print(f"\nBefore summarization:\n\n{args.text}")
    print("------------------------------------------------")
    print(f"After summarization:\n\n{summary}")
