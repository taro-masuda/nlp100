import torch
import os
import random
import numpy as np
from torch import nn
import torch.optim as optim
from single_layer_predict_71 import seed_everything, Net

if __name__ == "__main__":
    seed_everything()
    filedir_in = './data'
    modelpath = './data/torch_single_layer.pth'

    train_path = os.path.join(filedir_in, 'train.pt')
    x_train = torch.load(train_path)

    net = Net(in_shape=x_train.shape[1], out_shape=4)
    
    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_label = torch.load(train_label_path)
    y_label = torch.nn.functional.one_hot(y_label).to(torch.float)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), 
                        lr=0.001,
                        momentum=0.9)
    
    for epoch in range(100):
        optimizer.zero_grad()

        y_pred = net.forward(x_train)
        loss = criterion(y_pred, y_label)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print('epoch: {}, loss: {:.4f}'.format(epoch+1, loss))
    torch.save(net.state_dict(), modelpath)
    
    '''
    epoch: 1, loss: 0.5623
    epoch: 11, loss: 0.5619
    epoch: 21, loss: 0.5609
    epoch: 31, loss: 0.5599
    epoch: 41, loss: 0.5588
    epoch: 51, loss: 0.5577
    epoch: 61, loss: 0.5566
    epoch: 71, loss: 0.5555
    epoch: 81, loss: 0.5545
    epoch: 91, loss: 0.5535
    '''
    