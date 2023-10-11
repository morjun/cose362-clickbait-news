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
    #makeCsv()
    #tokenize("data.csv")
    train("tokenized.csv")

def tokenize(csvFile: str):
    pd.set_option('display.max_columns', None)

    fullFrame = pd.read_csv(csvFile)

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

    tokenFrame = pd.DataFrame({'title': tokenized_titles, 'content': tokenized_contents,'label': truncFrame['label']})
    tokenFrame.to_csv("tokenized.csv", index=False)

def makeCsv():
    dfs = [] # list of dataframes
    topDir = "data"
    for root, dirs, files in os.walk(topDir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    data = pandas.json_normalize(json.loads(f.read()))
                    print(f"Processing {root}\\{file}: {data['sourceDataInfo.newsTitle'][0]}")
                    dfs.append(data)
                    
    df = pandas.concat(dfs, ignore_index=True)
    
    print(df)
    df.to_csv("data.csv", index=False)
    


if __name__ == "__main__":
    main()