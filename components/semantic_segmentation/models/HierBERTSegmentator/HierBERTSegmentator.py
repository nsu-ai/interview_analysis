# -*- coding: utf-8 -*-

import logging
import json
from razdel import sentenize
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification
from .model.BertForTextSegmentation import BertForTextSegmentation


class HierBERTSegmentator:
    def __init__(self, model_path, sentence_encoder_path=None, tokenizer_path=None,
                 average_method="CLS", max_seq_length=128, use_cuda=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else sentence_encoder_path)
        self.model = BertForTextSegmentation.from_pretrained(model_path)
        self.average_method = average_method
        self.max_seq_length = max_seq_length
        self.use_cuda = use_cuda

        if sentence_encoder_path is not None:
            self.sentence_encoder = AutoModel.from_pretrained(sentence_encoder_path)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def predict(self, text: str):
        """

        """

        text_cleared = text.replace('\n', ' ')
        sentences = [s.text for s in sentenize(text_cleared)]

        if self.use_cuda:
            self.sentence_encoder.cuda()

        self.sentence_encoder.eval()
        self.model.eval()

        encoded_sentences = []
        for sent in sentences:
            with torch.no_grad():
                if self.average_method == "CLS":
                    inputs = self.tokenizer(sent, padding=True, truncation=True, max_length=self.max_seq_length,
                                       return_tensors="pt")

                    if self.use_cuda:
                        for i in inputs:
                            inputs[i] = inputs[i].cuda()

                    embedding = self.sentence_encoder(**inputs)[0][0][0].cpu().numpy().tolist()
                elif self.average_method == "mean":
                    inputs = self.tokenizer(sent, padding=True, truncation=True, max_length=self.max_seq_length,
                                       add_special_tokens=False, return_tensors="pt")

                    if self.use_cuda:
                        for i in inputs:
                            inputs[i] = inputs[i].cuda()

                    embedding = self.sentence_encoder(**inputs)
                    embedding = HierBERTSegmentator.mean_pooling(embedding, inputs['attention_mask'])[0].cpu().numpy().tolist()
                else:
                    raise NotImplementedError

            encoded_sentences.append(embedding)

        if self.use_cuda:
            self.sentence_encoder.cpu()

        model_input = np.array(encoded_sentences)

        # padded_model_input = np.zeros((self.max_seq_length, 768))
        # # assigning values
        # if model_input.shape[0] <= self.max_seq_length:
        #     padded_model_input[:model_input.shape[0], :] = model_input
        # else:
        #     padded_model_input[:self.max_seq_length, :] = model_input[:self.max_seq_length, :]
        #
        # attention_mask = [1] * min(model_input.shape[0], self.max_seq_length) + \
        #                  [0] * max(0, self.max_seq_length - model_input.shape[0])
        # attention_mask = torch.Tensor(np.array(attention_mask).reshape(1, len(attention_mask)))

        padded_model_input = model_input
        padded_model_input = torch.Tensor(padded_model_input.reshape(1, *padded_model_input.shape))
        if self.use_cuda:
            padded_model_input.cuda()
            #attention_mask.cuda()
            self.model.cuda()

        predictions = self.model(inputs_embeds=padded_model_input)[0]#, attention_mask=attention_mask)[0]
        predictions = torch.argmax(predictions, dim=2)[0].tolist()

        if self.use_cuda:
            self.model.cpu()

        print(predictions)

        segments = []
        cur_segment = []
        for i in range(len(sentences)):
            if predictions[i] == 0:
                cur_segment.append(sentences[i])
            else:
                segments.append(" ".join(cur_segment))
                cur_segment = [sentences[i]]
        if cur_segment:
            segments.append(" ".join(cur_segment))

        return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./", help="Directory with trained model")
    parser.add_argument("--sentence_encoder_path", type=str, default="./",
                        help="Directory with model for sentence encoding (BERT-based")
    parser.add_argument("--tokenizer_path", type=str, default="DeepPavlov/rubert-base-cased",
                        help="Directory with tokenizer for sentence encoding. If None, --sentence_encoder_path is used")
    parser.add_argument("--average_method", type=str, default="CLS",
                        help="Method for getting sentence embeddings from word embeddings")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum length of input sequence")
    parser.add_argument("--use_cuda", type=bool, default=False,
                        help="Use GPU to increase performance")
    parser.add_argument("--text", type=str, default="",
                        help="Input text for segmentation")

    args = parser.parse_args()

    segmentation_model = HierBERTSegmentator(model_path=args.model_path,
                                             sentence_encoder_path=args.sentence_encoder_path,
                                             tokenizer_path=args.tokenizer_path,
                                             average_method=args.average_method,
                                             max_seq_length=args.max_seq_length,
                                             use_cuda=False
                                            )

    segments = segmentation_model.predict(args.text)

    segments = "\n\n".join(segments)
    print(f"\nBefore segmentation:\n\n{args.text}")
    print("------------------------------------------------")
    print(f"After segmentation:\n\n{segments}")
