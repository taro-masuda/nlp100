import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

if __name__ == '__main__':
    
    train_path = './data/train.txt'
    val_path = './data/val.txt'
    test_path = './data/test.txt'

    fe = FeatureExtractor()

    df_train = pd.read_csv(train_path, sep='\t')
    train_fe_path = train_path.replace('.txt', '_features.npy')
    fe.extract_save_features(df_train, is_test=False,
                             filepath=train_fe_path)

    df_val = pd.read_csv(val_path, sep='\t')
    val_fe_path = val_path.replace('.txt', '_features.npy')
    fe.extract_save_features(df_val, is_test=True,
                             filepath=val_fe_path)

    df_test = pd.read_csv(test_path, sep='\t')
    test_fe_path = test_path.replace('.txt', '_features.npy')
    fe.extract_save_features(df_test, is_test=True,
                             filepath=test_fe_path)