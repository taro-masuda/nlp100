from re import X
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import random
import os
import pickle
from typing import Union
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def preprocess_label(label_path: str) -> Union[np.array, preprocessing.LabelEncoder]:
    df = pd.read_csv(label_path, sep='\t')
    le = preprocessing.LabelEncoder()
    le.fit(df['category'])
    return le.transform(df['category']), le

def train(x_train: np.array, y: np.array, C: float) -> LogisticRegression:
    clf = LogisticRegression(C=C).fit(x_train, y)
    return clf

if __name__ == '__main__':
    label_path = './data/train.txt'
    train_path = './data/train_features.npy'
    model_path = './data/model_logisticregression.pickle'
    
    val_path = './data/val.txt'
    test_path = './data/test.txt'

    fig_path = './data/acc_vs_param.png'

    y, le = preprocess_label(label_path=label_path)
    x_train = np.load(file=train_path)
    
    df_train = pd.read_csv(label_path, sep='\t')
    df_val = pd.read_csv(val_path, sep='\t')
    df_test = pd.read_csv(test_path, sep='\t')
    
    reguralization_parameters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    val_fe_path = './data/val_features.npy'
    test_fe_path = './data/test_features.npy'
    x_val = np.load(file=val_fe_path)
    x_test = np.load(file=test_fe_path)
    
    y_true_train = le.transform(df_train['category'])
    y_true_val = le.transform(df_val['category'])
    y_true_test = le.transform(df_test['category'])

    acc_tr, acc_val, acc_test = [], [], []
    for C in reguralization_parameters:
        clf = train(x_train=x_train, y=y, C=C)
        
        y_pred = clf.predict(x_train)
        acc_tr.append(accuracy_score(y_true_train, y_pred))        

        y_pred = clf.predict(x_val)
        acc_val.append(accuracy_score(y_true_val, y_pred))        

        y_pred = clf.predict(x_test)
        acc_test.append(accuracy_score(y_true_test, y_pred))        

    plt.plot(reguralization_parameters, acc_tr, 
            color='blue', marker='o', label='train')
    plt.plot(reguralization_parameters, acc_val, 
            color='green', marker='+', label='val')
    plt.plot(reguralization_parameters, acc_test, 
            color='red', marker='^', label='test')
    
    plt.xlabel('Regularization parameter C')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1)

    plt.savefig(fig_path)