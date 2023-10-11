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
    fullFrame = pd.read_csv("data.csv")
    print(fullFrame.shape)
    print(fullFrame.head())
    print(fullFrame.columns)
    content_bert = ["[CLS] " + str(sent) + " [SEP]" for sent in fullFrame['sourceDataInfo.newsContent']]
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in content_bert]

    print(tokenized_texts[0])



if __name__ == "__main__":
    main()