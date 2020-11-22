from transformers import pipeline
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def main():
    datapath = "data/"
    D_train= pd.read_csv(os.path.join(datapath, "challenge3/train.csv"))
    D_test = pd.read_csv(os.path.join(datapath, "challenge3/submission.csv"))
    # tokenizer 1=democrats, 0=Republican

    classifier = pipeline('sentiment-analysis')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)

    # pre-processing

    # feature 1
    author_train = set(D_train["Author"].unique())
    author_test = set(D_test["Author"].unique())
    author_map = D_train[["Author", "label"]].groupby("Author").mean()
    unknown = pd.DataFrame.from_dict({key: [-1] for key in unknown_authors}).T
    unknown.columns = ["label"]
    author_map = pd.concat([author_map, unknown])
    author_map = author_map.to_dict()["label"]
    D_test["feature1"] = D_test["Author"].map(author_map)
    D_train["feature1"] = D_train["Author"].map(author_map)
    D_test["feature1"][D_test["feature1"].isna()] = -1
    D_train["feature1"][D_train["feature1"].isna()] = -1

    # Body
    D_train["Body"]

if __name__ == "__main__":
    main()

