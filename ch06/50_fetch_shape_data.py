import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
import requests
import zipfile
import io

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def fetch_data(url: str, save_dir: str, filename: str) -> pd.DataFrame:
    req = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(req.content))
    zip_file.extractall(save_dir)

    df = pd.read_csv(os.path.join(save_dir, filename), sep='\t', header=None)
    return df

def shape_data(df: pd.DataFrame):#-> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df.columns = ['id', 'title', 'url', 'publisher', 'category', 'story', 'hostname', 'timestamp']
    df_cond = df[df['publisher'].isin(['Reuters' ,'Huffington Post',
                                        'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
    df_train, df_val_test = train_test_split(df_cond, test_size=0.2)
    df_val, df_test = train_test_split(df_val_test, test_size=0.5)
    return df_train, df_val, df_test

def fetch_shape_data(url: str, save_dir: str, filename: str):# -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df = fetch_data(url=url, save_dir=save_dir, filename=filename)
    train_df, valid_df, test_df = shape_data(df=df)
    return train_df, valid_df, test_df

if __name__ == '__main__':
    seed_everything(seed=42)
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip'
    save_dir = './data/'
    filename = 'newsCorpora.csv'
    df_train, df_val, df_test = fetch_shape_data(url=url, 
                                            save_dir=save_dir,
                                            filename=filename)
    df_train.to_csv(os.path.join(save_dir, 'train.txt'), sep='\t',
                    index=None)
    df_val.to_csv(os.path.join(save_dir, 'val.txt'), sep='\t',
                    index=None)
    df_test.to_csv(os.path.join(save_dir, 'test.txt'), sep='\t',
                    index=None)
    print('df_train record size:', len(df_train))
    print('df_val record size:', len(df_val))
    print('df_test record size:', len(df_test))
    '''
    df_train record size: 10672
    df_val record size: 1334
    df_test record size: 1334
    '''