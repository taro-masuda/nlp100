import torch
import os
import random
import numpy as np
from torch import nn

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Net(nn.Module):
    def __init__(self, in_shape: int, out_shape: int):
        super().__init__()
        self.fc = nn.Linear(300, 4, bias=True)
        nn.init.constant_(self.fc.bias.data, 0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    seed_everything()
    filedir_in = './data'

    train_path = os.path.join(filedir_in, 'train.pt')
    x_train = torch.load(train_path)

    net = Net(in_shape=x_train.shape[1], out_shape=4)
    y_pred = net.forward(x_train)

    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_label = torch.load(train_label_path)
    y_label = torch.nn.functional.one_hot(y_label).to(torch.float)
    
    loss = nn.BCELoss()
    output = loss(y_pred, y_label)
    print(output)
    output.backward()
    print(net.fc.weight.grad)
    assert net.fc.weight.grad.shape == net.fc.weight.shape
    '''
    tensor(0.5623, grad_fn=<BinaryCrossEntropyBackward>)
    tensor([[ 2.1097e-06, -1.6584e-06, -4.6904e-07,  ..., -1.1038e-06,
          1.3178e-06,  3.3685e-06],
        [-7.4744e-07,  5.0572e-07,  1.3389e-07,  ...,  4.3032e-07,
         -4.4466e-07, -1.1497e-06],
        [-7.0701e-07,  5.9917e-07,  1.7039e-07,  ...,  3.5053e-07,
         -4.5378e-07, -1.1540e-06],
        [-6.5526e-07,  5.5355e-07,  1.6476e-07,  ...,  3.2295e-07,
         -4.1934e-07, -1.0648e-06]])
    '''
    