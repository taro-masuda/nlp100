import torch
import os
import random
import numpy as np
from torch import nn
import torch.optim as optim
from single_layer_predict_71 import seed_everything
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

class Net(nn.Module):
    def __init__(self, in_shape: int, out_shape: int):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 128, bias=True)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(128, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(128, affine=True)
        self.fc3 = nn.Linear(128, out_shape, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def calc_acc(tensor_pred: torch.tensor, tensor_label: torch.tensor) -> float:
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
        return self.X[idx], self.y[idx]

def train(BATCH_SIZE=None, EPOCH=None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything()
    filedir_in = './data'
    ckpt_dir = './data/checkpoint/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    modelpath = os.path.join(ckpt_dir, 'torch_single_layer_79.pth')
    figpath ='./data/torch_training_79.png'

    train_path = os.path.join(filedir_in, 'train.pt')
    x_train = torch.load(train_path).to(device)

    val_path = os.path.join(filedir_in, 'val.pt')
    x_val = torch.load(val_path).to(device)

    net = Net(in_shape=x_train.shape[1], out_shape=4).to(device)
    
    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_tr_label = torch.load(train_label_path).to(device)
    y_tr_label = torch.nn.functional.one_hot(y_tr_label).to(torch.float)
    
    val_label_path = os.path.join(filedir_in, 'val_label.pt')
    y_val_label = torch.load(val_label_path).to(device)
    y_val_label = torch.nn.functional.one_hot(y_val_label).to(torch.float)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), 
                        lr=0.001,
                        momentum=0.9)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.xlabel('# epoch')
    plt.ylabel('loss')
    plt.subplot(2,1,2)
    plt.xlabel('# epoch')
    plt.ylabel('acc')

    dataset = TextDataset(X=x_train, y=y_tr_label)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    tr_loss, tr_acc, val_loss, val_acc = [],[],[],[]

    start_time = time.time()
    for epoch in range(EPOCH):
        for _ , (x_tr, y_tr) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred = net(x_tr)
            loss = criterion(y_pred, y_tr)
            loss.backward()
            optimizer.step()

        y_pred = net(x_train)
        loss = criterion(y_pred, y_tr_label)
        tr_loss.append(loss.item())
        acc = calc_acc(y_pred, y_tr_label)
        tr_acc.append(acc)

        if epoch % 10 == 0:
            plt.subplot(2,1,1)
            plt.scatter(epoch, loss.item(), color='blue', label='tr')
            plt.subplot(2,1,2)
            plt.scatter(epoch, acc, color='blue', label='tr')
        '''
        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': net.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            },
            modelpath.replace('.pth', 
                '_epoch' + str(epoch).zfill(3) + '.pth'))
        

        '''
        
        y_pred = net(x_val)
        loss = criterion(y_pred, y_val_label)
        val_loss.append(loss.item())
        acc = calc_acc(y_pred, y_val_label)
        val_acc.append(acc)

        
        if epoch % 10 == 0:
            plt.subplot(2,1,1)
            plt.scatter(epoch, loss.item(), color='red', label='val')
            plt.subplot(2,1,2)
            plt.scatter(epoch, acc, color='red', label='val')

    print('tr_loss: {}, tr_acc: {}, val_loss: {}, val_acc: {}'.format( 
        tr_loss[-1], tr_acc[-1], val_loss[-1], val_acc[-1]))

    print('Time per epoch:', 
        '{:.4f}[s]'.format((time.time() - start_time)/EPOCH),
        'Batch Size B =', BATCH_SIZE)
    torch.save(net.state_dict(), modelpath)
    plt.savefig(figpath)
    
if __name__ == "__main__":
    '''
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 465.27       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   34C    P0    25W / 250W |      0MiB / 16280MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    '''
    train(BATCH_SIZE=256, EPOCH=200) # 0.4107946026986507