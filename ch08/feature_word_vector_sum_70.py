import torch
from gensim.models.keyedvectors import KeyedVectors
import os
import pandas as pd
import numpy as np
import texthero as hero
import sklearn
from typing import Union

class FeatureExtractor:
    def __init__(self, filepath):
        self.word_vector = KeyedVectors.load_word2vec_format(filepath,
                                                    binary=True)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        self.custom_pipeline = [
            hero.preprocessing.fillna,
            hero.preprocessing.lowercase,
            hero.preprocessing.remove_digits,
            hero.preprocessing.remove_punctuation,
            hero.preprocessing.remove_diacritics,
            hero.preprocessing.remove_stopwords
        ]
        return hero.clean(df, self.custom_pipeline)

    def make_tensor(self, titles: list) -> torch.tensor:
        X = torch.zeros(len(titles), 300, dtype=torch.float)
        for i, title in enumerate(titles):
            words = title.split(' ')
            skipped_cnt = 0
            for word in words:
                if word not in self.word_vector:
                    skipped_cnt += 1
                    continue
                X[i, :] += self.word_vector[word]
            X /= max(len(words) - skipped_cnt, 1) # avoid zero division
        return X

    def make_feature_pipeline(self, df: pd.DataFrame) -> torch.tensor:
        df['clean_title'] = self.preprocess(df)
        titles = df['clean_title'].tolist()
        return self.make_tensor(titles)

    def label_fit(self, df: pd.DataFrame) -> None:
        self.le = sklearn.preprocessing.LabelEncoder()
        self.le.fit(df['category'])
        
    def label_transform(self, df: pd.DataFrame) -> np.array:
        return self.le.transform(df['category'])

if __name__ == "__main__":
    filepath_bin = './data/GoogleNews-vectors-negative300.bin'
    filedir_text = './data'
    filedir_out = './data'

    #==========MAKE X==========
    fe = FeatureExtractor(filepath=filepath_bin)

    train_path = os.path.join(filedir_text, 'train.txt')
    df_train = pd.read_csv(train_path, sep='\t')
    
    x_train = fe.make_feature_pipeline(df_train['title'])
    train_path_out = os.path.join(filedir_out, 'train.pt')
    print(x_train.shape)
    torch.save(x_train, train_path_out)

    val_path = os.path.join(filedir_text, 'val.txt')
    df_val = pd.read_csv(val_path, sep='\t')
    
    x_val = fe.make_feature_pipeline(df_val['title'])
    val_path_out = os.path.join(filedir_out, 'val.pt')
    print(x_val.shape)
    torch.save(x_val, val_path_out)

    test_path = os.path.join(filedir_text, 'test.txt')
    df_test = pd.read_csv(test_path, sep='\t')
    
    x_test = fe.make_feature_pipeline(df_test['title'])
    test_path_out = os.path.join(filedir_out, 'test.pt')
    print(x_test.shape)
    torch.save(x_test, test_path_out)
    '''
    torch.Size([10672, 300])
    torch.Size([1334, 300])
    torch.Size([1334, 300])
    '''
    #==========MAKE Y==========
    fe.label_fit(df_train)
    
    y_train = fe.label_transform(df_train)
    y_train_torch = torch.tensor(y_train)
    train_path_out = os.path.join(filedir_out, 'train_label.pt')
    torch.save(y_train_torch, train_path_out)
    
    y_val = fe.label_transform(df_val)
    y_val_torch = torch.tensor(y_val)
    val_path_out = os.path.join(filedir_out, 'val_label.pt')
    torch.save(y_val_torch, val_path_out)
    
    y_test = fe.label_transform(df_test)
    y_test_torch = torch.tensor(y_test)
    test_path_out = os.path.join(filedir_out, 'test_label.pt')
    torch.save(y_test_torch, test_path_out)
    print(y_train.shape, y_val.shape, y_test.shape)
    '''
    (10672,) (1334,) (1334,)
    '''