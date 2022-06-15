# -*- coding: utf-8 -*-

import logging
import json
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# TODO:
#  1. Переписать в предикте все на батчевый подход
#  2. Перейти с words-to-pred к работе на уровне токенов, так надежнее
#  3. Добавить клининг текста
#  4. Подумать об оверлапе с двух сторон
#  5. Подумать, как мерджить предсказания оверлапа
#  6. Вернуть буквы Й и Ё
#  7. Попробовать улучшить предсказание точек за счет введения меток вида '.UO' вместо 'O.'


class BertPunctRestorer:
    def __init__(self, model_path, labels_path, words_per_pred=200):
        self.words_per_pred = words_per_pred
        self.overlap_words = 30

        with open(labels_path) as labels_file:
            self.labels = json.load(labels_file)
        self.id2label = dict(enumerate(self.labels))

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

    def predict(self, text: str):
        """
        Performs punctuation restoration on arbitrarily large text.
        Detects if input is not English, if non-English was detected terminates predictions.
        Overrride by supplying `lang='en'`

        Args:
            - text (str): Text to punctuate, can be few words to as large as you want.


        lang argument may be returned in a future
        """
        words = text.replace('\n', ' ').split(" ")
        words = [w for w in words if w]

        splits = self.split_on_chunks(words)

        predictions, tokens = [], []
        for split in splits:
            pred, tok = self._predict(" ".join(split))
            predictions.append(pred)
            tokens.append(tok)

        joined_tokens, joined_predictions = self.join_chunks(splits, tokens, predictions)

        punct_text = self.apply_predictions(joined_tokens, joined_predictions)

        return punct_text

    def _predict(self, input_slice):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(input_slice)))
        inputs = self.tokenizer.encode(input_slice, return_tensors="pt")
        predictions = torch.argmax(self.model(inputs)[0], dim=2)[0].tolist()
        predictions = [self.id2label[pred] for pred in predictions]

        # join bpe split tokens
        new_tokens, new_predictions = [], []
        for token, pred in zip(tokens, predictions):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
                if new_predictions[-1][-1] == "O" and pred[-1] != 0:
                    new_predictions[-1] = new_predictions[-1][0] + pred[-1]
                # Возможно, если часть слова с заглавной, то все слово с заглавной
            else:
                new_predictions.append(pred)
                new_tokens.append(token)

        return new_predictions, new_tokens

    def split_on_chunks(self, words):
        """
        Splits text into predefined slices of overlapping text with indexes (offsets)
        that tie-back to original text.
        This is done to bypass 512 token limit on transformer models by sequentially
        feeding chunks of < 512 toks.
        Example output:
        [{...}, {"text": "...", 'start_idx': 31354, 'end_idx': 32648}, {...}]
        """
        resp = []
        start = 0

        while True:
            # words in the chunk and the overlapping portion
            words_split = words[start:start + self.words_per_pred + self.overlap_words]

            # Break loop if no more words
            if not words_split:
                break

            resp.append(words_split)

            start = start + self.words_per_pred

        logging.info(f"Sliced transcript into {len(resp)} slices.")
        return resp

    def join_chunks(self, words_splits, tokens, predictions):
        """
        Given a full text, predictions of each slice combines predictions into a single text again.
        Performs validation whether text was combined correctly
        """
        joined_predictions = []
        joined_tokens = []

        for i in range(len(words_splits)):
            # assert words_splits[i] == tokens[i][1:-1], f"{words_splits[i]}\n{tokens[i][1:-1]}"

            # Process [CLS] and [SEP] tokens
            offset = 0 if len(joined_tokens) == 0 else self.overlap_words
            joined_predictions.extend(predictions[i][1 + offset:-1])
            joined_tokens.extend(tokens[i][1 + offset:-1])

        return joined_tokens, joined_predictions

    def apply_predictions(self, joined_tokens, joined_predictions):
        """
        Given a list of Predictions from the model, applies the predictions to text,
        thus punctuating it.
        """
        text = ""
        for token, pred in zip(joined_tokens, joined_predictions):

            if pred[0] == "U":
                punctuated_word = token.capitalize()
            else:
                punctuated_word = token

            if pred[-1] != "O":
                punctuated_word += pred[-1]

            text += punctuated_word + " "
        text = text.strip()

        # Append trailing period if doesnt exist
        if len(text) > 0 and text[-1].isalnum():
            text += "."
        return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./",
                        help="Directory with trained model")
    parser.add_argument("--labels_path", type=str, default="./labels.json",
                        help="Text file with labels")
    parser.add_argument("--text", type=str, default="",
                        help="Input text for punctuation restoration")

    args = parser.parse_args()

    punct_model = BertPunctRestorer(model_path=args.model_path, labels_path=args.labels_path)

    punctuated = punct_model.predict(args.text)

    print(f"\nBefore punctuation restoration:\n\n{args.text}")
    print("------------------------------------------------")
    print(f"After punctuation restoration:\n\n{punctuated}")
