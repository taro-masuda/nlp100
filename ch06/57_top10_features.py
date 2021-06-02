import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from typing import Union
import pickle

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def extract_features(self, df: pd.DataFrame, 
                        is_test: bool) -> np.array:
        if is_test:
            X = self.vectorizer.transform(df['title']).toarray()
        else:
            X = self.vectorizer.fit_transform(df['title']).toarray()
        return X

    def extract_save_features(self, 
                            df: pd.DataFrame,
                            is_test: bool, 
                            filepath: str) -> None:
        array = self.extract_features(df, is_test=is_test)
        np.save(file=filepath, arr=array, allow_pickle=True)

def preprocess_label(label_path: str) -> Union[np.array, preprocessing.LabelEncoder]:
    df = pd.read_csv(label_path, sep='\t')
    le = preprocessing.LabelEncoder()
    le.fit(df['category'])
    return le.transform(df['category']), le

if __name__ == '__main__':
    
    train_path = './data/train.txt'
    val_path = './data/val.txt'
    test_path = './data/test.txt'

    fe = FeatureExtractor()

    df_train = pd.read_csv(train_path, sep='\t')
    train_fe_path = train_path.replace('.txt', '_features.npy')
    fe.extract_save_features(df_train, is_test=False,
                             filepath=train_fe_path)

    model_path = './data/model_logisticregression.pickle'
    y, le = preprocess_label(label_path=train_path)
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    feature_names = fe.vectorizer.get_feature_names()
    for i, coef in enumerate(clf.coef_):
        print('===TOP10 weights for class {}==='.format(
            le.classes_[i]
        ))
        top10_indices = coef.argsort()[-10:][::-1]
        for idx in top10_indices:
            print(feature_names[idx])
    for i, coef in enumerate(clf.coef_):
        print('===LEAST10 weights for class {}==='.format(
            le.classes_[i]
        ))
        top10_indices = coef.argsort()[:10]
        for idx in top10_indices:
            print(feature_names[idx])
    '''
    ===TOP10 weights for class b===
    china
    fed
    stocks
    bank
    ecb # ECB（European Central Bank）は、欧州中央銀行のことを指します。
    euro
    oil
    update
    ukraine
    yellen # Janet Louise Yellenは、アメリカ合衆国の政治家、経済学者。
    ===TOP10 weights for class e===
    kardashian # kim kardashianは、アメリカ合衆国のソーシャライト、リアリティ番組パーソナリティ、モデル、女優。
    chris
    star
    kim
    miley # Miley Cyrusは、アメリカ合衆国出身のシンガーソングライター、女優、音楽プロデューサー、慈善家。
    her
    cyrus
    she
    film
    movie
    ===TOP10 weights for class m===
    ebola
    cancer
    study
    drug
    fda # FDAとは、アメリカ食品医薬品局（Food and Drug Administration）の略称
    mers # 中東呼吸器症候群
    cases
    could
    outbreak
    virus
    ===TOP10 weights for class t===
    google
    facebook
    apple
    microsoft
    climate
    gm
    nasa
    mobile
    tesla
    comcast
    ===LEAST10 weights for class b===
    and
    the
    her
    video
    ebola
    she
    microsoft
    study
    kardashian
    google
    ===LEAST10 weights for class e===
    update
    us
    google
    china
    says
    facebook
    gm
    ceo
    apple
    study
    ===LEAST10 weights for class m===
    gm
    facebook
    google
    amazon
    apple
    sales
    ceo
    twitter
    climate
    billion
    ===LEAST10 weights for class t===
    stocks
    fed
    drug
    her
    american
    ecb
    cancer
    kardashian
    york
    shares
    '''