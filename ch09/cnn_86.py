import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import os
from map_to_id_80 import IDMapping
from rnn_prediction_81 import seed_everything, FeatureExtractor
from torch.utils.data import Dataset, DataLoader
import time
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

def calc_acc(tensor_pred, tensor_label: torch.tensor) -> float:
    y_te_pred = torch.argmax(tensor_pred, dim=1)
    y_label = torch.argmax(tensor_label, dim=1)
    acc = (y_te_pred == y_label).sum().item() / y_label.shape[0]
  
    assert acc >= 0 and acc <= 1
    return acc

class CNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                output_size: int,
                seq_len: int):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                padding_mode='zeros'
                                )
        self.pool1d = nn.MaxPool1d(kernel_size=3, stride=seq_len, padding=0)
        self.fc = nn.Linear(
                        in_features=in_channels,
                        out_features=output_size,
                        bias=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.tensor):
        N = x.shape[0]
        x = self.conv1d(x)
        x = self.pool1d(x) # N(samples) x C(n_dim(300)) x 1
        x = torch.squeeze(x) # N x C(n_dim) 
        x = x.reshape([N, -1])
        x = self.fc(x) # N x 4
        x = self.softmax(x) # 1 x 4
        return x

class WordEmbeddingExtractor(FeatureExtractor):
    def __init__(self, filepath):
        self.word_vector = KeyedVectors.load_word2vec_format(filepath,
                                                    binary=True)
    def make_feature(self, titles: list, dic: dict) -> torch.tensor:
        X = []
        for i, title in enumerate(titles):
            words = title.split(' ')
            l = []
            for word in words:
                if word in self.word_vector:
                    l.append(self.word_vector[word][:50])
                else:
                    l.append(np.zeros(50,))
            X.append(torch.tensor(l, dtype=torch.float)) # n_samples x seq_len x n_dim
        X = nn.utils.rnn.pad_sequence(X)
        X = X.permute(1, 2, 0) # n_samples(N) x n_dim(Cin) x seq_len(L)
        return X    

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def predict(config: dict):
    if torch.cuda.is_available():
      device = torch.device('cuda:0') 
      torch.cuda.empty_cache()
    else:
      device = torch.device('cpu')

    seed_everything()

    filedir_in = './data'
    filepath_bin = './data/GoogleNews-vectors-negative300.bin'
    train_path = os.path.join(filedir_in, 'train.txt')
    val_path = os.path.join(filedir_in, 'val.txt')

    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_tr_label = torch.load(train_label_path).to(device)
    y_tr_label = torch.nn.functional.one_hot(y_tr_label).to(torch.float)

    val_label_path = os.path.join(filedir_in, 'val_label.pt')
    y_val_label = torch.load(val_label_path).to(device)
    y_val_label = torch.nn.functional.one_hot(y_val_label).to(torch.float)

    input_size = config['input_size']
    output_size = config['output_size']

    fe = WordEmbeddingExtractor(filepath=filepath_bin)
    df_train = pd.read_csv(train_path, sep='\t')
    idmapping = IDMapping()
    idmapping.words_to_ids(df_train)
    dic = idmapping.dic
    x_train = fe.make_feature_pipeline(df=df_train['title'],
                                        dic=dic).to(device)
    df_val = pd.read_csv(val_path, sep='\t')
    x_val = fe.make_feature_pipeline(df=df_val['title'],
                                        dic=dic).to(device)
    batch_size_val = x_val.shape[1]

    dataset = TextDataset(X=x_train, y=y_tr_label)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    #print(x_train.shape)
    net = CNN(in_channels=x_train.shape[1], out_channels=x_train.shape[1],
                output_size=config['output_size'],
                seq_len=x_train.shape[2]).to(device)

    output = net.forward(x_train).to(device)
    y_pred = output
    print(y_pred[:5, :])


if __name__ == '__main__':
    config = {
        'batch_size': 1,
        'input_size': 50,
        'output_size': 4
    }
    predict(config=config)
    '''
    tensor([[0.2218, 0.2626, 0.2475, 0.2681],
        [0.2333, 0.2471, 0.2360, 0.2836],
        [0.2399, 0.2456, 0.2286, 0.2859],
        [0.2404, 0.2453, 0.2362, 0.2782],
        [0.2396, 0.2443, 0.2339, 0.2821]], grad_fn=<SliceBackward>)
    '''