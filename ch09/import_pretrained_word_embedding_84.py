import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import os
from map_to_id_80 import IDMapping
from rnn_prediction_81 import seed_everything, FeatureExtractor, RNN
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

class RNNEmbedding(RNN):
    def forward(self, x: torch.tensor, h_0: torch.tensor):
        x, h_T = self.rnn(x, h_0)
        x = self.fc(x)
        x = self.softmax(x)
        return x, h_T

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
                    l.append(self.word_vector[word])
                else:
                    l.append(np.zeros(300,))
            X.append(torch.tensor(l, dtype=torch.float)) # n_samples x seq_len x
        X = nn.utils.rnn.pad_sequence(X)
        return X    

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[:, idx], self.y[idx]

def train(config: dict):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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

    hidden_size = 50
    input_size = 300
    output_size = 4

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

    net = RNNEmbedding(input_size=input_size, 
            hidden_size=hidden_size,
            output_size=output_size,
            n_vocab=len(dic)).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(),
                    lr=0.001,
                    momentum=0.9)

    start_time = time.time()
    for epoch in range(config['epoch']):
        for x_tr, y_tr in dataloader:
            #print(x_tr.shape)
            optimizer.zero_grad()
            x_tr = x_tr.permute(1, 0, 2)
            output, h_T = net(x=x_tr, h_0=torch.zeros(1, x_tr.shape[1], hidden_size).to(device))
            y_pred = output[-1, :, :]
            loss = criterion(y_pred, y_tr)
            tr_loss = loss.item()
            tr_acc = calc_acc(y_pred, y_tr)
            loss.backward()
            optimizer.step()

        output, h_T = net(x_train, h_0=torch.zeros(1, x_train.shape[1], hidden_size).to(device))
        y_pred = output[-1, :, :]
        loss = criterion(y_pred, y_tr_label)
        tr_loss = loss.item()
        tr_acc = calc_acc(y_pred, y_tr_label)

        output, h_T = net(x=x_val, h_0=torch.zeros(1, batch_size_val, hidden_size).to(device))
        y_pred = output[-1, :, :]
        loss = criterion(y_pred, y_val_label)
        val_loss = loss.item()
        val_acc = calc_acc(y_pred, y_val_label)

        print('epoch: {}, tr_loss: {:.4f}, tr_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(
            epoch+1, 
            tr_loss,
            tr_acc,
            val_loss,
            val_acc
            )
        )
    print('Time per epoch: ', (time.time() - start_time)/config['epoch'], '[s]')

if __name__ == '__main__':
    config = {
        'epoch': 100,
        'batch_size': 32,
    }
    train(config=config)
    '''
    epoch: 1, tr_loss: 0.5687, tr_acc: 0.4252, val_loss: 0.5694, val_acc: 0.4040
    .
    .
    .
    epoch: 100, tr_loss: 0.4880, tr_acc: 0.4253, val_loss: 0.4954, val_acc: 0.4040
    Time per epoch:  0.2349664044380188 [s]
    '''