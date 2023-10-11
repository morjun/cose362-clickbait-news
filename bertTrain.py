import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

# pip install scikit-learn

import pandas as pd
import numpy as np
import random
import time
import os
import json
import datetime


def main():
    # makeCsv()
    tokenize("data.csv")
    # train("tokenized.csv")


def tokenize(csvFile: str):
    pd.set_option("display.max_columns", None)

    fullFrame = pd.read_csv(csvFile)

    print(fullFrame.shape)
    print(fullFrame.head())
    print(fullFrame.columns)

    truncFrame = fullFrame.loc[
        :,
        [
            "sourceDataInfo.newsTitle",
            "sourceDataInfo.newsContent",
            "labeledDataInfo.clickbaitClass",
        ],
    ]
    truncFrame = truncFrame.rename(
        columns={
            "sourceDataInfo.newsTitle": "title",
            "sourceDataInfo.newsContent": "content",
            "labeledDataInfo.clickbaitClass": "label",
        }
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased", do_lower_case=False
    )

    truncFrame["content"] = (
        truncFrame["content"]
        .str.replace(r"\\n", " ", regex=True)
        .replace(r"\\[^a-z]", "", regex=True)
        .replace("[.,!?-]", "")
    )
    truncFrame["title"] = (
        truncFrame["title"]
        .str.replace(r"\\n", " ", regex=True)
        .replace(r"\\[^a-z]", "", regex=True)
        .replace("[.,!?-]", "")
    )

    # truncFrame['content'] = ["[CLS] " + str(sent) + " [END]" for sent in truncFrame['content']]
    # truncFrame['title'] = ["[CLS] " + str(sent) + " [END]" for sent in truncFrame['title']]
    # [SEP]는 전체 내용의 끝이 아니라 문장과 문장 사이에 붙여야 한다.
    # https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
    contentBatchList = []
    titleBatchList = []

    for content in truncFrame["content"]:
        contentBatchList.append(preprocessSentence(content, tokenizer))
    for title in truncFrame["title"]:
        titleBatchList.append(preprocessSentence(title, tokenizer))

    tokenizedContents = [tokenizer.tokenize(sent) for sent in truncFrame["content"]]
    tokenizedTitles = [tokenizer.tokenize(sent) for sent in truncFrame["title"]]

    # tokenizedContentsIds = [
    #     tokenizer.convert_tokens_to_ids(x) for x in tokenized_contents
    # ]
    # tokenizedTitlesIds = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_titles]

    tokenFrame = pd.DataFrame(
        {
            "titleToken": tokenizedTitles,
            "titleTokenId": [x['input_ids'] for x in titleBatchList],
            "titleAttentionMask": [x['attention_mask'] for x in titleBatchList],
            "contentToken": tokenizedContents,
            "contentTokenId": [x['input_ids'] for x in contentBatchList],
            "contentAttentionMask": [x['attention_mask'] for x in contentBatchList],
            "label": truncFrame["label"],
        }
    )
    tokenFrame.to_csv("tokenized.csv", index=False)


def preprocessSentence(input_text: str, tokenizer: BertTokenizer):
    # tokenizer.encode_plus
    """
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
        - input_ids: list of token ids
        - token_type_ids: list of token type ids
        - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    """
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def makeCsv():
    dfs = []  # list of dataframes
    topDir = "data"
    for root, dirs, files in os.walk(topDir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    data = pd.json_normalize(json.loads(f.read()))
                    print(
                        f"Processing {root}\\{file}: {data['sourceDataInfo.newsTitle'][0]}"
                    )
                    dfs.append(data)

    df = pd.concat(dfs, ignore_index=True)

    print(df)
    df.to_csv("data.csv", index=False)


def train(tokenCsv: str):
    NotImplemented


if __name__ == "__main__":
    main()
