import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from tqdm import trange 

# pip install scikit-learn

import pandas as pd
import numpy as np
import random
import time
import os
import json
import datetime

OPTION = "allInOne"

def main():
    # makeCsv()
    # tokenId, attentionMask, labels = tokenize("data.csv")

    tokenId = torch.load("tokenId.pt")
    attentionMask = torch.load("attentionMask.pt")
    labels = torch.load("label.pt")

    trainDataLoader, valDataLoader = splitData(tokenId=tokenId, attentionMask=attentionMask, label=labels)
    train(trainDataLoader, valDataLoader)


def concatContentsAndTitle(df: pd.DataFrame):
    df["content"] = df["content"].str.cat(df["title"], sep=" ")
    return df

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
    ] #제목, 내용, 레이블만 추출

    truncFrame = truncFrame.rename(
        columns={
            "sourceDataInfo.newsTitle": "title",
            "sourceDataInfo.newsContent": "content",
            "labeledDataInfo.clickbaitClass": "label",
        }
    )

    if OPTION == "allInOne":
        truncFrame = concatContentsAndTitle(truncFrame)

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased", do_lower_case=False,
    )

    truncFrame["content"] = (
        truncFrame["content"]
        .str.replace(r"\\n", " ", regex=True)
        .replace(r"\\[^a-z]", "", regex=True)
        .replace("[.,!?-]", "")
    ) #escape sequence 및 특수문자 제거

    if OPTION != "allInOne":
        truncFrame["title"] = (
            truncFrame["title"]
            .str.replace(r"\\n", " ", regex=True)
            .replace(r"\\[^a-z]", "", regex=True)
            .replace("[.,!?-]", "")
        )

    contentBatchList = []
    titleBatchList = []

    tokenId = []
    attentionMasks = []

    for content in truncFrame["content"]:
        encodingDict = preprocessSentence(content, tokenizer)
        tokenId.append(encodingDict["input_ids"])
        attentionMasks.append(encodingDict["attention_mask"])


    if OPTION != "allInOne":
        for title in truncFrame["title"]:
            titleBatchList.append(preprocessSentence(title, tokenizer))

    tokenizedContents = [tokenizer.tokenize(sent) for sent in truncFrame["content"]]

    tokenizedTitles = []
    if OPTION != "allInOne":
        tokenizedTitles = [tokenizer.tokenize(sent) for sent in truncFrame["title"]]

    tokenId = torch.cat(tokenId, dim = 0)
    attentionMasks = torch.cat(attentionMasks, dim = 0)
    label = torch.tensor(truncFrame["label"].values)

    torch.save(tokenId, "tokenId.pt")
    torch.save(attentionMasks, "attentionMask.pt")
    torch.save(label, "label.pt")

    return tokenId, attentionMasks, label

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
        max_length=512, # BERT가 지원하는 최대 token 수
        padding='max_length',
        truncation=True,
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


def splitData(tokenId: torch.Tensor, attentionMask: torch.Tensor, label: torch.Tensor):
    VAL_RATIO = 0.1
    BATCH_SIZE = 32

    trainIdx, valIdx = train_test_split(
        np.arange(len(label)), test_size=VAL_RATIO, shuffle=True, stratify=label
    )

    trainSet = TensorDataset(
        tokenId[trainIdx], attentionMask[trainIdx], label[trainIdx]
    ) # 주어진 token들 중 training set과 validation set에 해당하는 인덱스 값들을 각각 구한다.
    valSet = TensorDataset(tokenId[valIdx], attentionMask[valIdx], label[valIdx])

    trainDataLoader = DataLoader(
        trainSet, sampler=RandomSampler(trainSet), batch_size=BATCH_SIZE
    )
    valDataLoader = DataLoader(
        valSet, sampler=SequentialSampler(valSet), batch_size=BATCH_SIZE
    )

    return trainDataLoader, valDataLoader


def train(trainDataLoader: DataLoader, valDataLoader: DataLoader):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    model.cuda()

    try:
        device = torch.device("cuda")
    except Exception as err:
        print("cuda is not available")
        exit(1)


    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    epochs = 2

    for _ in trange(epochs, desc="Epoch"):
        # ========== Training ==========

        # Set model to training mode
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(trainDataLoader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass
            train_output = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            # Backward pass
            train_output.loss.backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += train_output.loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        # ========== Validation ==========

        # Set model to evaluation mode
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                eval_output = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )
            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != "nan":
                val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != "nan":
                val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != "nan":
                val_specificity.append(b_specificity)

        print("\n\t - Train loss: {:.4f}".format(tr_loss / nb_tr_steps))
        print(
            "\t - Validation Accuracy: {:.4f}".format(sum(val_accuracy) / len(val_accuracy))
        )
        print(
            "\t - Validation Precision: {:.4f}".format(
                sum(val_precision) / len(val_precision)
            )
            if len(val_precision) > 0
            else "\t - Validation Precision: NaN"
        )
        print(
            "\t - Validation Recall: {:.4f}".format(sum(val_recall) / len(val_recall))
            if len(val_recall) > 0
            else "\t - Validation Recall: NaN"
        )
        print(
            "\t - Validation Specificity: {:.4f}\n".format(
                sum(val_specificity) / len(val_specificity)
            )
            if len(val_specificity) > 0
            else "\t - Validation Specificity: NaN"
        )


def b_tp(preds, labels):
    """Returns True Positives (TP): count of correct predictions of actual class 1"""
    return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])


def b_fp(preds, labels):
    """Returns False Positives (FP): count of wrong predictions of actual class 1"""
    return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])


def b_tn(preds, labels):
    """Returns True Negatives (TN): count of correct predictions of actual class 0"""
    return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])


def b_fn(preds, labels):
    """Returns False Negatives (FN): count of wrong predictions of actual class 0"""
    return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])


def b_metrics(preds, labels):
    """
    Returns the following metrics:
      - accuracy    = (TP + TN) / N
      - precision   = TP / (TP + FP)
      - recall      = TP / (TP + FN)
      - specificity = TN / (TN + FP)
    """
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else "nan"
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else "nan"
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else "nan"
    return b_accuracy, b_precision, b_recall, b_specificity


if __name__ == "__main__":
    main()
