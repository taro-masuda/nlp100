import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import texthero as hero
import sklearn
import pandas as pd
import torch

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords
    ]
    return hero.clean(df, custom_pipeline)
        

class IDMapping:
    def __init__(self):
        self.dic = {}
    def words_to_ids(self, df: pd.DataFrame) -> dict:
        df['clean_title'] = preprocess(df['title'])
        titles = df['clean_title'].tolist()
        for title in titles:
            for word in title.split(' '):
                if word in self.dic:
                    self.dic[word] += 1
                else:
                    self.dic[word] = 1
        idx = 1
        for key in sorted(self.dic, key=self.dic.get, reverse=True):
            if self.dic[key] == 1:
                self.dic[key] = 0
            else:
                self.dic[key] = idx
                idx += 1

    def make_feature_pipeline(self, df: pd.DataFrame) -> torch.tensor:
        df['clean_title'] = self.preprocess(df)
        titles = df['clean_title'].tolist()
        return self.make_tensor(titles)

if __name__ == '__main__':
    train_path = './data/train.txt'
    df = pd.read_csv(train_path, sep='\t')

    id_mapping = IDMapping()
    id_mapping.words_to_ids(df)

    for key in sorted(id_mapping.dic, key=id_mapping.dic.get, reverse=True)[:5]:
        print(key, id_mapping.dic[key])
    
    for key in sorted(id_mapping.dic, key=id_mapping.dic.get, reverse=False)[:5]:
        print(key, id_mapping.dic[key])
    '''
    crews 7414
    rubin 7413
    scorecard 7412
    upfront 7411
    newsday 7410
    gardening 0
    odin 0
    hopefully 0
    merlin 0
    microphone 0
    '''