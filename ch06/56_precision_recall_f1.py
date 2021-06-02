import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_label(label_path: str):
    df = pd.read_csv(label_path, sep='\t')
    le = preprocessing.LabelEncoder()
    le.fit(df['category'])
    return le.transform(df['category']), le

if __name__ == '__main__':
    model_path = './data/model_logisticregression.pickle'
    train_path = './data/train.txt'
    test_fe_path = './data/test_features.npy'
    test_path = './data/test.txt'
    fig_path = './data/confusion_matrix.png'

    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    _, le = preprocess_label(label_path=train_path)
    df_test = pd.read_csv(test_path, sep='\t')
    
    x_test = np.load(file=test_fe_path)
    y_true = le.transform(df_test['category'])
    y_pred = clf.predict(x_test)
    print(le.classes_)
    result = precision_recall_fscore_support(y_true, y_pred, average=None,
                                    labels=le.transform(le.classes_))
    
    print('Precision:', result[0])
    print('Recall:', result[1])
    print('F1:', result[2])
    '''
    ['b' 'e' 'm' 't']
    Precision: [0.86       0.87478261 0.94       0.8440367 ]
    Recall: [0.93818182 0.97669903 0.43518519 0.57142857]
    F1: [0.8973913  0.92293578 0.59493671 0.68148148]
    '''