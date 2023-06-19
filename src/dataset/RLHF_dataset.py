import sys
import os
import json
import pprint
import torch
import transformers

import numpy as np
import pandas as pd

from copy import deepcopy

from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class RLHFDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        tokenizer,
        max_len: int,
    ):
        self.datapath = datapath
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id

        self.data = []
        raw_data = json.load(open(self.datapath, "r"))
        for index, entry in enumerate(raw_data):
            
            guid = entry["guid"] if "guid" in entry.keys() else None
            
            question = entry["question"]
            answer = entry["answer"]

            # Some of the entries have an explanation, some don't
            if "explanation" in entry.keys():
                explanation = entry["explanation"]
                if explanation != None:
                    question = question + " " + explanation

            # Handling the multiple choices
            if "choices" in entry.keys():
                choices = entry["choices"]
                if choices != None:
                    for choice in choices:
                        question = question + " " + choice

            question_tokenized = tokenizer.encode_plus(
                question,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
            )

            answer_tokenized = tokenizer.encode_plus(
                answer,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
            )

            question_input_ids = question_tokenized["input_ids"]
            question_attention_mask = question_tokenized["attention_mask"]

            answer_input_ids = answer_tokenized["input_ids"]
            answer_attention_mask = answer_tokenized["attention_mask"]

            self.data.append(
                {
                    "question_input_ids": question_input_ids,
                    "question_attention_mask": question_attention_mask,
                    "answer_input_ids": answer_input_ids,
                    "answer_attention_mask": answer_attention_mask,
                    "question": question,
                    "answer": answer,
                    "guid": guid
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return deepcopy(self.data[index])  # gonna fix it later on if needed
