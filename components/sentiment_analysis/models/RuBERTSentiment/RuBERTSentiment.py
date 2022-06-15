# -*- coding: utf-8 -*-

import logging
import json
import argparse
from transformers import BertTokenizerFast, AutoModelForSequenceClassification
import torch


class RuBERTSentiment:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self, text: str):
        """
        """

        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = int(torch.argmax(predicted, dim=1).numpy()[0])

        # Labels ID to labels
        id2label = {0: "NEUTRAL",
                    1: "POSITIVE",
                    2: "NEGATIVE"}
        predicted = id2label[predicted]

        return predicted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="blanchefort/rubert-base-cased-sentiment"),
    parser.add_argument("--text", type=str, default="",
                        help="Input text for punctuation restoration")

    args = parser.parse_args()

    model = RuBERTSentiment(model_path=args.model_path)

    sentiment = model.predict(args.text)

    print(f"\nBefore sentiment analysis:\n\n{args.text}")
    print("------------------------------------------------")
    print(f"After sentiment analysis:\n\n{sentiment}")
