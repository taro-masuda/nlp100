import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

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
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    _, le = preprocess_label(label_path=train_path)
    df_test = pd.read_csv(test_path, sep='\t')
    
    x_test = np.load(file=test_fe_path)
    y_pred = clf.predict(x_test)
    
    y_true = le.transform(df_test['category'])
    print('Accuracy score:', accuracy_score(y_true, y_pred))
    '''
    Accuracy score: 0.8680659670164917
    '''