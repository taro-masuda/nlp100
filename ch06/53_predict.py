import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import random
import os

def preprocess_label(label_path: str) -> (np.array, preprocessing.LabelEncoder):
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
    y, le = preprocess_label(label_path=label_path)
    x_train = np.load(file=train_path)
    clf = train(x_train=x_train, y=y)
    
    val_path = './data/val_features.npy'
    x_val = np.load(file=val_path)
    cl_label = clf.predict(x_val)
    print('The model predicted as :', le.inverse_transform(np.array(cl_label)))
    proba = clf.predict_proba(x_val)
    print('The probabilities are:', proba)
    '''
    The model predicted as : ['e' 'e' 'b' ... 'b' 'b' 'e']
    The probabilities are: [[0.03265459 0.91556784 0.02185072 0.02992685]
    [0.01747122 0.9613966  0.01121398 0.00991821]
    [0.63848099 0.20266014 0.05108513 0.10777373]
    ...
    [0.717289   0.11202372 0.05043762 0.12024966]
    [0.58238952 0.21762238 0.07239006 0.12759804]
    [0.10441898 0.81488677 0.02948049 0.05121375]]
    '''