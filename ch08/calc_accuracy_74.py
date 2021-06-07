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
    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_tr_label = torch.load(train_label_path)

    net = Net(in_shape=x_train.shape[1], out_shape=4)
    net.load_state_dict(torch.load(modelpath))
    
    y_tr_pred = net(x_train)
    y_tr_pred = torch.argmax(y_tr_pred, dim=1)
    acc_tr = (y_tr_pred == y_tr_label).sum().item() / y_tr_label.shape[0]
    print(acc_tr)

    test_path = os.path.join(filedir_in, 'test.pt')
    x_test = torch.load(test_path)
    test_label_path = os.path.join(filedir_in, 'test_label.pt')
    y_te_label = torch.load(test_label_path)

    y_te_pred = net(x_test)
    y_te_pred = torch.argmax(y_te_pred, dim=1)
    acc_te = (y_te_pred == y_te_label).sum().item() / y_te_label.shape[0]
    print(acc_te)
    '''
    0.4252248875562219
    0.411544227886057
    '''
    