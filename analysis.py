import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
#pip install scikit-learn

import pandas as pd
import numpy as np
import random
import time
import datetime

def main():
    pd.set_option('display.max_columns', None)

    fullFrame = pd.read_csv("data.csv")

    print(fullFrame.shape)
    print(fullFrame.head())
    print(fullFrame.columns)

    truncFrame = fullFrame.loc[:, ['sourceDataInfo.newsTitle', 'sourceDataInfo.newsContent', 'labeledDataInfo.clickbaitClass']]
    truncFrame = truncFrame.rename(columns={"sourceDataInfo.newsTitle": "title", "sourceDataInfo.newsContent": "content", "labeledDataInfo.clickbaitClass": "label"})

    truncFrame['content'] = truncFrame['content'].str.replace(r'\\n', ' ', regex=True)
    truncFrame['content'] = truncFrame['content'].str.replace(r'\\[^a-z]', '', regex=True)
    truncFrame['title'] = truncFrame['title'].str.replace(r'\\n', ' ', regex=True)
    truncFrame['title'] = truncFrame['title'].str.replace(r'\\[^a-z]', '', regex=True)

    content_bert = ["[CLS] " + str(sent) + " [SEP]" for sent in truncFrame['content']]
    title_bert = ["[CLS] " + str(sent) + " [SEP]" for sent in truncFrame['title']]

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_contents = [tokenizer.tokenize(sent) for sent in content_bert]
    tokenized_titles = [tokenizer.tokenize(sent) for sent in title_bert]

    print(tokenized_contents[0])


if __name__ == "__main__":
    main()