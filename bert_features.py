from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import h5py


def model():
    input1 = tf.keras.Input(shape=(768,), dtype=tf.float32, name='x1')
    input2 = tf.keras.Input(shape=(768,), dtype=tf.float32, name='x2')
    x1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(input1)
    x2 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(input2)
    x = tf.tensordot(x1, x2)
    x = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(x)
    return tf.keras.Model(inputs=[input1, input2], outputs=x)

datapath = "data/"
D_train= pd.read_csv(os.path.join(datapath, "challenge3/train.csv"))
D_test = pd.read_csv(os.path.join(datapath, "challenge3/submission.csv"))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)


# tokenize
X = D_train["Body"][~D_train["Body"].isna()].to_list()
batch = len(X)//1000
for t in range(len(X)//batch):
    with h5py.File(f'../data/challenge3/body_train/batch_{t:04d}.hdf5', 'w') as f:
        dset = f.create_dataset("Dataset", (batch, 768))
        _tokens = tokenizer(X[t * batch: (t + 1) * batch], padding=True, truncation=True, return_tensors="tf")
        dset[t * batch: (t + 1) * batch, :] = bert(_tokens)[1]

X = D_train["Post"].to_list()
batch = len(X)//10000
with h5py.File('../data/challenge3/post_train_features.hdf5', 'w') as f:
    dset = f.create_dataset("Dataset", (len(X), 768))
    for t in range(len(X)//batch):
        _tokens = tokenizer(X[t * batch: (t + 1) * batch], padding=True, truncation=True, return_tensors="tf")
        dset[t * batch: (t + 1) * batch, :] = bert(_tokens)[1]
    for t in range(10000*batch, len(X_body)):
        _tokens = tokenizer(X[t], padding=True, truncation=True, return_tensors="tf")
        dset[t, :] = bert(_tokens)[1]


X = D_test["Post"][~D_test["Post"].isna()].to_list()
batch = len(X)//10000
with h5py.File('../data/challenge3/post_test_features.hdf5', 'w') as f:
    dset = f.create_dataset("Dataset", (len(X), 768))
    for t in range(len(X)//batch):
        _tokens = tokenizer(X[t * batch: (t + 1) * batch], padding=True, truncation=True, return_tensors="tf")
        dset[t * batch: (t + 1) * batch, :] = bert(_tokens)[1]
    for t in range(10000*batch, len(X_body)):
        _tokens = tokenizer(X[t], padding=True, truncation=True, return_tensors="tf")
        dset[t, :] = bert(_tokens)[1]


X = D_test["Body"][~D_test["Body"].isna()].to_list()
batch = len(X)//10000
with h5py.File('../data/challenge3/body_test_features.hdf5', 'w') as f:
    dset = f.create_dataset("Dataset", (len(X), 768))
    for t in range(len(X)//batch):
        _tokens = tokenizer(X[t * batch: (t + 1) * batch], padding=True, truncation=True, return_tensors="tf")
        dset[t * batch: (t + 1) * batch, :] = bert(_tokens)[1]
    for t in range(10000*batch, len(X_body)):
        _tokens = tokenizer(X[t], padding=True, truncation=True, return_tensors="tf")
        dset[t, :] = bert(_tokens)[1]
