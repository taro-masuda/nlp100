import torch
import os
import random
import numpy as np
from torch import nn
import torch.optim as optim
from single_layer_predict_71 import seed_everything, Net
import matplotlib.pyplot as plt
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
        return self.X[idx], self.y[idx]

def train(BATCH_SIZE=None, EPOCH=None):
    seed_everything()
    filedir_in = './data'
    ckpt_dir = './data/checkpoint/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    modelpath = os.path.join(ckpt_dir, 'torch_single_layer_77.pth')
    figpath ='./data/torch_training_77.png'

    train_path = os.path.join(filedir_in, 'train.pt')
    x_train = torch.load(train_path)

    val_path = os.path.join(filedir_in, 'val.pt')
    x_val = torch.load(val_path)

    net = Net(in_shape=x_train.shape[1], out_shape=4)
    
    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_tr_label = torch.load(train_label_path)
    y_tr_label = torch.nn.functional.one_hot(y_tr_label).to(torch.float)
    
    val_label_path = os.path.join(filedir_in, 'val_label.pt')
    y_val_label = torch.load(val_label_path)
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

        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': net.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            },
            modelpath.replace('.pth', 
                '_epoch' + str(epoch).zfill(3) + '.pth'))

        plt.subplot(2,1,1)
        plt.scatter(epoch, loss.item(), color='blue', label='tr')
        plt.subplot(2,1,2)
        plt.scatter(epoch, acc, color='blue', label='tr')
        
        y_pred = net(x_val)
        loss = criterion(y_pred, y_val_label)
        val_loss.append(loss.item())
        acc = calc_acc(y_pred, y_val_label)
        val_acc.append(acc)

        plt.subplot(2,1,1)
        plt.scatter(epoch, loss.item(), color='red', label='val')
        plt.subplot(2,1,2)
        plt.scatter(epoch, acc, color='red', label='val')
    print('Time per epoch:', 
        '{:.4f}[s]'.format((time.time() - start_time)/EPOCH),
        'Batch Size B =', BATCH_SIZE)
    plt.subplot(2,1,1)
    plt.legend()
    plt.subplot(2,1,2)
    plt.legend()
    torch.save(net.state_dict(), modelpath)
    plt.savefig(figpath)
    
if __name__ == "__main__":
    train(BATCH_SIZE=16, EPOCH=100) # Time per epoch: 0.2812[s] Batch Size B = 16
    train(BATCH_SIZE=8, EPOCH=100) # Time per epoch: 0.5002[s] Batch Size B = 8
    train(BATCH_SIZE=4, EPOCH=100) # Time per epoch: 0.9159[s] Batch Size B = 4
    train(BATCH_SIZE=2, EPOCH=100) # Time per epoch: 1.7519[s] Batch Size B = 2
    train(BATCH_SIZE=1, EPOCH=100) # Time per epoch: 3.3733[s] Batch Size B = 1