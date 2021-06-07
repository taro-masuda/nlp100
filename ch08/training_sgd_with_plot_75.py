import torch
import os
import random
import numpy as np
from torch import nn
import torch.optim as optim
from single_layer_predict_71 import seed_everything, Net
import matplotlib.pyplot as plt

def calc_acc(tensor_pred, tensor_label: torch.tensor) -> float:
    y_te_pred = torch.argmax(tensor_pred, dim=1)
    y_label = torch.argmax(tensor_label, dim=1)
    acc = (y_te_pred == y_label).sum().item() / y_label.shape[0]
  
    assert acc >= 0 and acc <= 1
    return acc

if __name__ == "__main__":
    seed_everything()
    filedir_in = './data'
    modelpath = './data/torch_single_layer.pth'
    figpath ='./data/torch_training.png'

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

    tr_loss, tr_acc, val_loss, val_acc = [],[],[],[]
    for epoch in range(100):
        optimizer.zero_grad()

        y_pred = net.forward(x_train)
        loss = criterion(y_pred, y_tr_label)
        loss.backward()
        optimizer.step()
        tr_loss.append(loss.item())
        acc = calc_acc(y_pred, y_tr_label)
        tr_acc.append(acc)

        plt.subplot(2,1,1)
        plt.scatter(epoch, loss.item(), color='blue', label='tr')
        plt.subplot(2,1,2)
        plt.scatter(epoch, acc, color='blue', label='tr')
        
        y_pred = net.forward(x_val)
        loss = criterion(y_pred, y_val_label)
        val_loss.append(loss.item())
        acc = calc_acc(y_pred, y_val_label)
        val_acc.append(acc)

        plt.subplot(2,1,1)
        plt.scatter(epoch, loss.item(), color='red', label='val')
        plt.subplot(2,1,2)
        plt.scatter(epoch, acc, color='red', label='val')
    
    plt.subplot(2,1,1)
    plt.legend()
    plt.subplot(2,1,2)
    plt.legend()
    torch.save(net.state_dict(), modelpath)
    plt.savefig(figpath)
    