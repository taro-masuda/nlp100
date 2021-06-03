import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import random
import os
import pickle
from typing import Union
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import wandb
import xgboost as xgb

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def preprocess_label(label_path: str) -> Union[np.array, preprocessing.LabelEncoder]:
    df = pd.read_csv(label_path, sep='\t')
    le = preprocessing.LabelEncoder()
    le.fit(df['category'])
    return le.transform(df['category']), le

def train(x_train: np.array, y: np.array, C: float,
            model_type: str):
    if model_type == 'LR':
        clf = LogisticRegression(C=C).fit(x_train, y)
    elif model_type == 'NB':
        clf = MultinomialNB(alpha=C).fit(x_train, y)
    elif model_type == 'RF':
        clf = RandomForestClassifier(n_estimators=int(C*100))
        clf.fit(x_train, y)
    else:
        raise ValueError
    return clf

if __name__ == '__main__':
    seed_everything()

    # wandb.init(project='visualize-sklearn')
    
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
    models = ['RF', 'NB', 'LR']#['NB', 'LR', 'SVC'] # LogisticRegression, SupportVectorClassifier, NaiveBayes

    for model_type in models:
        for C in reguralization_parameters:
            clf = train(x_train=x_train, y=y, C=C, model_type=model_type)
            
            y_pred = clf.predict(x_val)
            y_probas = clf.predict_proba(x_val)
            acc = accuracy_score(y_true_val, y_pred)
            acc_val.append(acc)        

            print(acc, model_type, C)
            '''
            wandb.log({
                'model': model_type, 
                'val_acc': acc,
                'hyperparam': C 
            })
            '''
    '''
    0.6964017991004497 RF 0.01
    0.7518740629685158 RF 0.03
    0.8148425787106447 RF 0.1
    0.808095952023988 RF 0.3
    0.8140929535232384 RF 1
    0.8245877061469266 RF 3
    0.8215892053973014 RF 10
    0.9227886056971514 NB 0.01
    0.9197901049475262 NB 0.03
    0.9122938530734632 NB 0.1
    0.8973013493253373 NB 0.3
    0.8268365817091454 NB 1
    0.7856071964017991 NB 3
    0.7773613193403298 NB 10
    0.7331334332833583 LR 0.01
    0.7466266866566716 LR 0.03
    0.7698650674662668 LR 0.1
    0.835832083958021 LR 0.3
    0.8928035982008995 LR 1
    0.9145427286356822 LR 3
    0.9220389805097451 LR 10
    '''