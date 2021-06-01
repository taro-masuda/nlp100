import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import random
import os
import pickle
from typing import Union

def preprocess_label(label_path: str) -> Union[np.array, preprocessing.LabelEncoder]:
    df = pd.read_csv(label_path, sep='\t')
    le = preprocessing.LabelEncoder()
    le.fit(df['category'])
    return le.transform(df['category']), le

def train(x_train: np.array, y: np.array) -> LogisticRegression:
    clf = LogisticRegression().fit(x_train, y)
    return clf

if __name__ == '__main__':
    label_path = './data/train.txt'
    train_path = './data/train_features.npy'
    model_path = './data/model_logisticregression.pickle'

    y, le = preprocess_label(label_path=label_path)
    x_train = np.load(file=train_path)
    clf = train(x_train=x_train, y=y)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print('Training done.')