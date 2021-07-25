import torch
from torch import nn
import random
import os
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import texthero as hero
import pandas as pd
from map_to_id_80 import IDMapping

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class FeatureExtractor:
    def __init__(self, filepath):
        #self.word_vector = KeyedVectors.load_word2vec_format(filepath,
        #                                            binary=True)
        pass
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
    
    def make_feature(self, titles: list, dic: dict) -> torch.tensor:
        X = []
        for i, title in enumerate(titles):
            words = title.split(' ')
            l = []
            for word in words:
                if word in dic:
                    l.append(dic[word])
                else:
                    l.append(0)
            X.append(torch.tensor(l, dtype=torch.int)) # n_samples x seq_len x
        X = nn.utils.rnn.pad_sequence(X)
        return X

    def make_feature_pipeline(self, df: pd.DataFrame,
                            dic: dict) -> torch.tensor:
        df['clean_title'] = self.preprocess(df)
        titles = df['clean_title'].tolist()
        return self.make_feature(titles=titles, dic=dic)

class RNN(nn.Module):
    def __init__(self, input_size: int, 
                hidden_size: int, 
                output_size: int,
                n_vocab: int):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, input_size)
        self.rnn = nn.RNN(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        nonlinearity='tanh',
                        bias=True,
                        bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_size,
                        out_features=output_size,
                        bias=True)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x: torch.tensor, h_0: torch.tensor):
        x = self.embedding(x) # seq_len x n_samples x n_dim(embed)
        #x = x.permute(1, 0, 2)
        x, h_T = self.rnn(x, h_0)
        x = self.fc(x)
        x = self.softmax(x)
        return x, h_T

if __name__ == '__main__':
    seed_everything()
    filepath_bin = './data/GoogleNews-vectors-negative300.bin'
    filedir_text = './data'
    train_path = os.path.join(filedir_text, 'train.txt')
    
    hidden_size = 50
    input_size = 300
    output_size = 4

    fe = FeatureExtractor(filepath=filepath_bin)
    df_train = pd.read_csv(train_path, sep='\t')
    idmapping = IDMapping()
    idmapping.words_to_ids(df_train)
    dic = idmapping.dic
    x_train = fe.make_feature_pipeline(df=df_train['title'],
                                        dic=dic)

    batch_size = x_train.shape[1]

    net = RNN(input_size=input_size, 
            hidden_size=hidden_size,
            output_size=output_size,
            n_vocab=len(dic))
    output, h_T = net(x_train, h_0=torch.zeros(1, batch_size, hidden_size))
    print(output.shape)
    print(output[-1, 0, :])
    '''
    torch.Size([419, 10672, 4])
    tensor([0.1590, 0.3011, 0.2402, 0.2997], grad_fn=<SliceBackward>)
    '''