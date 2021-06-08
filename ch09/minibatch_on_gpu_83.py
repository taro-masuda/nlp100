import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import os
from map_to_id_80 import IDMapping
from rnn_prediction_81 import seed_everything, FeatureExtractor, RNN
from torch.utils.data import Dataset, DataLoader
import time

def calc_acc(tensor_pred, tensor_label: torch.tensor) -> float:
    y_te_pred = torch.argmax(tensor_pred, dim=1)
    y_label = torch.argmax(tensor_label, dim=1)
    acc = (y_te_pred == y_label).sum().item() / y_label.shape[0]
  
    assert acc >= 0 and acc <= 1
    return acc

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

    fe = FeatureExtractor(filepath=filepath_bin)
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

    net = RNN(input_size=input_size, 
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
            optimizer.zero_grad()
            x_tr = x_tr.permute(1, 0)
            output, h_T = net.forward(x=x_tr, h_0=torch.zeros(1, x_tr.shape[1], hidden_size).to(device))
            y_pred = output[-1, :, :]
            loss = criterion(y_pred, y_tr)
            tr_loss = loss.item()
            tr_acc = calc_acc(y_pred, y_tr)
            loss.backward()
            optimizer.step()

        output, h_T = net.forward(x_train, h_0=torch.zeros(1, x_train.shape[1], hidden_size).to(device))
        y_pred = output[-1, :, :]
        loss = criterion(y_pred, y_tr_label)
        tr_loss = loss.item()
        tr_acc = calc_acc(y_pred, y_tr_label)

        output, h_T = net.forward(x=x_val, h_0=torch.zeros(1, batch_size_val, hidden_size).to(device))
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
    print('Time per epoch: ', (time.time() - start_time)/config['epoch'])

if __name__ == '__main__':
    '''
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 465.27       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   41C    P0    27W / 250W |      0MiB / 16280MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    '''
    config = {
        'epoch': 100,
        'batch_size': 32,
    }
    train(config=config)
    '''
    epoch: 1, tr_loss: 0.5184, tr_acc: 0.3961, val_loss: 0.5212, val_acc: 0.4010
    epoch: 2, tr_loss: 0.4955, tr_acc: 0.4252, val_loss: 0.5022, val_acc: 0.4048
    epoch: 3, tr_loss: 0.4910, tr_acc: 0.4252, val_loss: 0.4976, val_acc: 0.4048
    epoch: 4, tr_loss: 0.4898, tr_acc: 0.4252, val_loss: 0.4968, val_acc: 0.4048
    epoch: 5, tr_loss: 0.4894, tr_acc: 0.4252, val_loss: 0.4964, val_acc: 0.4093
    epoch: 6, tr_loss: 0.4890, tr_acc: 0.4252, val_loss: 0.4964, val_acc: 0.4048
    epoch: 7, tr_loss: 0.4887, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4093
    epoch: 8, tr_loss: 0.4888, tr_acc: 0.4252, val_loss: 0.4961, val_acc: 0.4093
    epoch: 9, tr_loss: 0.4880, tr_acc: 0.4252, val_loss: 0.4963, val_acc: 0.4048
    epoch: 10, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4048
    epoch: 11, tr_loss: 0.4877, tr_acc: 0.4252, val_loss: 0.4953, val_acc: 0.4093
    epoch: 12, tr_loss: 0.4887, tr_acc: 0.3961, val_loss: 0.4957, val_acc: 0.4010
    epoch: 13, tr_loss: 0.4875, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4048
    epoch: 14, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4958, val_acc: 0.4048
    epoch: 15, tr_loss: 0.4885, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4048
    epoch: 16, tr_loss: 0.4881, tr_acc: 0.4252, val_loss: 0.4956, val_acc: 0.4093
    epoch: 17, tr_loss: 0.4887, tr_acc: 0.3961, val_loss: 0.4956, val_acc: 0.4010
    epoch: 18, tr_loss: 0.4881, tr_acc: 0.4252, val_loss: 0.4954, val_acc: 0.4160
    epoch: 19, tr_loss: 0.4879, tr_acc: 0.4252, val_loss: 0.4955, val_acc: 0.4093
    epoch: 20, tr_loss: 0.4888, tr_acc: 0.4252, val_loss: 0.4970, val_acc: 0.4048
    epoch: 21, tr_loss: 0.4889, tr_acc: 0.4252, val_loss: 0.4977, val_acc: 0.4048
    epoch: 22, tr_loss: 0.4881, tr_acc: 0.4252, val_loss: 0.4961, val_acc: 0.4048
    epoch: 23, tr_loss: 0.4877, tr_acc: 0.4252, val_loss: 0.4958, val_acc: 0.4048
    epoch: 24, tr_loss: 0.4889, tr_acc: 0.3961, val_loss: 0.4959, val_acc: 0.4010
    epoch: 25, tr_loss: 0.4888, tr_acc: 0.3961, val_loss: 0.4959, val_acc: 0.4010
    epoch: 26, tr_loss: 0.4877, tr_acc: 0.3961, val_loss: 0.4954, val_acc: 0.4010
    epoch: 27, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4048
    epoch: 28, tr_loss: 0.4888, tr_acc: 0.4252, val_loss: 0.4970, val_acc: 0.4048
    epoch: 29, tr_loss: 0.4890, tr_acc: 0.4252, val_loss: 0.4981, val_acc: 0.4048
    epoch: 30, tr_loss: 0.4877, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4048
    epoch: 31, tr_loss: 0.4885, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4093
    epoch: 32, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4958, val_acc: 0.4048
    epoch: 33, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4048
    epoch: 34, tr_loss: 0.4879, tr_acc: 0.3961, val_loss: 0.4954, val_acc: 0.4010
    epoch: 35, tr_loss: 0.4884, tr_acc: 0.3961, val_loss: 0.4957, val_acc: 0.4010
    epoch: 36, tr_loss: 0.4885, tr_acc: 0.4252, val_loss: 0.4961, val_acc: 0.4093
    epoch: 37, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4048
    epoch: 38, tr_loss: 0.4884, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4085
    epoch: 39, tr_loss: 0.4881, tr_acc: 0.4252, val_loss: 0.4967, val_acc: 0.4048
    epoch: 40, tr_loss: 0.4889, tr_acc: 0.4252, val_loss: 0.4976, val_acc: 0.4048
    epoch: 41, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4048
    epoch: 42, tr_loss: 0.4881, tr_acc: 0.4252, val_loss: 0.4961, val_acc: 0.4048
    epoch: 43, tr_loss: 0.4884, tr_acc: 0.4252, val_loss: 0.4970, val_acc: 0.4048
    epoch: 44, tr_loss: 0.4885, tr_acc: 0.4252, val_loss: 0.4964, val_acc: 0.4048
    epoch: 45, tr_loss: 0.4875, tr_acc: 0.4252, val_loss: 0.4958, val_acc: 0.4048
    epoch: 46, tr_loss: 0.4874, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4048
    epoch: 47, tr_loss: 0.4877, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4048
    epoch: 48, tr_loss: 0.4879, tr_acc: 0.3961, val_loss: 0.4955, val_acc: 0.4010
    epoch: 49, tr_loss: 0.4879, tr_acc: 0.4252, val_loss: 0.4968, val_acc: 0.4048
    epoch: 50, tr_loss: 0.4880, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4048
    epoch: 51, tr_loss: 0.4883, tr_acc: 0.3961, val_loss: 0.4958, val_acc: 0.4010
    epoch: 52, tr_loss: 0.4880, tr_acc: 0.3961, val_loss: 0.4956, val_acc: 0.4010
    epoch: 53, tr_loss: 0.4875, tr_acc: 0.4252, val_loss: 0.4958, val_acc: 0.4048
    epoch: 54, tr_loss: 0.4881, tr_acc: 0.4252, val_loss: 0.4968, val_acc: 0.4048
    epoch: 55, tr_loss: 0.4876, tr_acc: 0.4252, val_loss: 0.4955, val_acc: 0.4093
    epoch: 56, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4955, val_acc: 0.4093
    epoch: 57, tr_loss: 0.4883, tr_acc: 0.4252, val_loss: 0.4963, val_acc: 0.4048
    epoch: 58, tr_loss: 0.4885, tr_acc: 0.4252, val_loss: 0.4974, val_acc: 0.4048
    epoch: 59, tr_loss: 0.4875, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4048
    epoch: 60, tr_loss: 0.4882, tr_acc: 0.3961, val_loss: 0.4956, val_acc: 0.4010
    epoch: 61, tr_loss: 0.4889, tr_acc: 0.3961, val_loss: 0.4960, val_acc: 0.4010
    epoch: 62, tr_loss: 0.4886, tr_acc: 0.3961, val_loss: 0.4958, val_acc: 0.4010
    epoch: 63, tr_loss: 0.4881, tr_acc: 0.4252, val_loss: 0.4968, val_acc: 0.4048
    epoch: 64, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4956, val_acc: 0.4048
    epoch: 65, tr_loss: 0.4893, tr_acc: 0.3961, val_loss: 0.4962, val_acc: 0.4010
    epoch: 66, tr_loss: 0.4888, tr_acc: 0.4252, val_loss: 0.4961, val_acc: 0.4093
    epoch: 67, tr_loss: 0.4886, tr_acc: 0.4252, val_loss: 0.4962, val_acc: 0.4085
    epoch: 68, tr_loss: 0.4879, tr_acc: 0.4252, val_loss: 0.4961, val_acc: 0.4048
    epoch: 69, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4955, val_acc: 0.4093
    epoch: 70, tr_loss: 0.4878, tr_acc: 0.3961, val_loss: 0.4954, val_acc: 0.4010
    epoch: 71, tr_loss: 0.4888, tr_acc: 0.3961, val_loss: 0.4959, val_acc: 0.4010
    epoch: 72, tr_loss: 0.4887, tr_acc: 0.4252, val_loss: 0.4963, val_acc: 0.4048
    epoch: 73, tr_loss: 0.4893, tr_acc: 0.4252, val_loss: 0.4966, val_acc: 0.4093
    epoch: 74, tr_loss: 0.4884, tr_acc: 0.4252, val_loss: 0.4957, val_acc: 0.4085
    epoch: 75, tr_loss: 0.4876, tr_acc: 0.4252, val_loss: 0.4957, val_acc: 0.4048
    epoch: 76, tr_loss: 0.4883, tr_acc: 0.4252, val_loss: 0.4965, val_acc: 0.4048
    epoch: 77, tr_loss: 0.4875, tr_acc: 0.4252, val_loss: 0.4959, val_acc: 0.4048
    epoch: 78, tr_loss: 0.4883, tr_acc: 0.3961, val_loss: 0.4958, val_acc: 0.4010
    epoch: 79, tr_loss: 0.4888, tr_acc: 0.4252, val_loss: 0.4964, val_acc: 0.4048
    epoch: 80, tr_loss: 0.4889, tr_acc: 0.4252, val_loss: 0.4967, val_acc: 0.4048
    epoch: 81, tr_loss: 0.4888, tr_acc: 0.4252, val_loss: 0.4965, val_acc: 0.4093
    epoch: 82, tr_loss: 0.4880, tr_acc: 0.4252, val_loss: 0.4963, val_acc: 0.4048
    epoch: 83, tr_loss: 0.4879, tr_acc: 0.4252, val_loss: 0.4968, val_acc: 0.4048
    epoch: 84, tr_loss: 0.4880, tr_acc: 0.4252, val_loss: 0.4961, val_acc: 0.4048
    epoch: 85, tr_loss: 0.4889, tr_acc: 0.4252, val_loss: 0.4973, val_acc: 0.4048
    epoch: 86, tr_loss: 0.4890, tr_acc: 0.4252, val_loss: 0.4969, val_acc: 0.4048
    epoch: 87, tr_loss: 0.4883, tr_acc: 0.4252, val_loss: 0.4963, val_acc: 0.4048
    epoch: 88, tr_loss: 0.4876, tr_acc: 0.4252, val_loss: 0.4957, val_acc: 0.4048
    epoch: 89, tr_loss: 0.4878, tr_acc: 0.4252, val_loss: 0.4957, val_acc: 0.4093
    epoch: 90, tr_loss: 0.4889, tr_acc: 0.3961, val_loss: 0.4961, val_acc: 0.4010
    epoch: 91, tr_loss: 0.4883, tr_acc: 0.4252, val_loss: 0.4960, val_acc: 0.4093
    epoch: 92, tr_loss: 0.4896, tr_acc: 0.4252, val_loss: 0.4971, val_acc: 0.4093
    epoch: 93, tr_loss: 0.4899, tr_acc: 0.4252, val_loss: 0.4980, val_acc: 0.4048
    epoch: 94, tr_loss: 0.4889, tr_acc: 0.4252, val_loss: 0.4968, val_acc: 0.4048
    epoch: 95, tr_loss: 0.4892, tr_acc: 0.3961, val_loss: 0.4963, val_acc: 0.4010
    epoch: 96, tr_loss: 0.4882, tr_acc: 0.4252, val_loss: 0.4956, val_acc: 0.4093
    epoch: 97, tr_loss: 0.4877, tr_acc: 0.4252, val_loss: 0.4955, val_acc: 0.4093
    epoch: 98, tr_loss: 0.4876, tr_acc: 0.4252, val_loss: 0.4954, val_acc: 0.4093
    epoch: 99, tr_loss: 0.4885, tr_acc: 0.3961, val_loss: 0.4957, val_acc: 0.4010
    epoch: 100, tr_loss: 0.4877, tr_acc: 0.4252, val_loss: 0.4957, val_acc: 0.4048
    Time per epoch:  0.28687769889831544
    '''