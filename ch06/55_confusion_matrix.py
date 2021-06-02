import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix
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
    plot_confusion_matrix(estimator=clf, X=x_test, y_true=y_true,
                        display_labels=le.classes_)
    # plt.show()
    plt.savefig(fname=fig_path)