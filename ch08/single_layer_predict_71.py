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
    
    #assert torch.equal(net.fc(torch.zeros_like(x_train)) \
    #    , torch.zeros(x_train.shape[0], 4))
    output = net.forward(x_train)

    print(output[0, :])
    print(output[0:4, :])
    print(output.shape)
    '''
    tensor([0.2500, 0.2500, 0.2500, 0.2500], grad_fn=<SliceBackward>)
    tensor([[0.2500, 0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500, 0.2500]], grad_fn=<SliceBackward>)
    torch.Size([10672, 4])
    '''
    