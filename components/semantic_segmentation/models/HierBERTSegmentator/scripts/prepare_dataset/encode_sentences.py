import os
import json
import logging
import csv
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import BertModel, BertPreTrainedModel, BertConfig, BertForTokenClassification


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_sentences(input_dir, output_dir, model_path, tokenizer_path=None, average_method="CLS",
                     use_cuda=False, max_seq_length=512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else model_path)
    sentence_encoder = AutoModel.from_pretrained(model_path)

    if use_cuda:
        sentence_encoder.cuda()
    sentence_encoder.eval()

    for file_name in os.listdir(input_dir):
        output_fil_name = f"{'.'.join(file_name.split('.')[:-1])}_{average_method}.{file_name.split('.')[-1]}"

        with open(os.path.join(input_dir, file_name), encoding="utf-8") as input_file, \
                open(os.path.join(output_dir, output_fil_name), "w", newline="", encoding="utf-8") as output_file:

            reader = csv.reader(input_file)
            writer = csv.writer(output_file, delimiter="\t")

            for row in tqdm(reader, f"Processing {file_name}"):
                new_row = ""

                if row:
                    with torch.no_grad():
                        if average_method == "CLS":
                            inputs = tokenizer(row[0], padding=True, truncation=True, max_length=max_seq_length,
                                               return_tensors="pt")

                            if use_cuda:
                                for i in inputs:
                                    inputs[i] = inputs[i].cuda()

                            embedding = sentence_encoder(**inputs)[0][0][0].cpu().numpy().tolist()
                        elif average_method == "mean":
                            inputs = tokenizer(row[0], padding=True, truncation=True, max_length=max_seq_length,
                                               add_special_tokens=False, return_tensors="pt")

                            if use_cuda:
                                for i in inputs:
                                    inputs[i] = inputs[i].cuda()

                            embedding = sentence_encoder(**inputs)
                            embedding = mean_pooling(embedding, inputs['attention_mask'])[0].cpu().numpy().tolist()
                        else:
                            raise NotImplementedError
                    new_row = [embedding, row[1]]

                writer.writerow(new_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="./",
                        help="Directory with sentence-based datasets")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory for prepared embeddings-based dateset")
    parser.add_argument("--model_path", type=str, default="./",
                        help="Path to model using for embeddings generation")
    parser.add_argument("--tokenizer_path", type=str, default="DeepPavlov/rubert-base-cased",
                        help="Path to tokenizer using for embeddings generation. It can be name of pretrained model")
    parser.add_argument("--average_method", type=str, default="CLS",
                        help="Method for sentence embeddings generation from BERT token embeddings. "
                             "'CLS' - using only [CLS] embedding, 'mean' - using average of all non-special tokens")
    parser.add_argument("--use_cuda", type=bool, default=False,
                        help="Using GPU accelerating")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequent legth for truncating and padding")

    args = parser.parse_args()

    encode_sentences(args.input_dir, args.output_dir, model_path=args.model_path,
                     tokenizer_path=args.tokenizer_path, average_method=args.average_method,
                     use_cuda=args.use_cuda, max_seq_length=args.max_seq_length)
