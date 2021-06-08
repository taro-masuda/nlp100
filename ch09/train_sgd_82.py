import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import os
from map_to_id_80 import IDMapping
from rnn_prediction_81 import seed_everything, FeatureExtractor, RNN

def calc_acc(tensor_pred, tensor_label: torch.tensor) -> float:
    y_te_pred = torch.argmax(tensor_pred, dim=1)
    y_label = torch.argmax(tensor_label, dim=1)
    acc = (y_te_pred == y_label).sum().item() / y_label.shape[0]
  
    assert acc >= 0 and acc <= 1
    return acc

def train(config: dict):
    seed_everything()
    filedir_in = './data'
    filepath_bin = './data/GoogleNews-vectors-negative300.bin'
    train_path = os.path.join(filedir_in, 'train.txt')
    val_path = os.path.join(filedir_in, 'val.txt')

    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_tr_label = torch.load(train_label_path)
    y_tr_label = torch.nn.functional.one_hot(y_tr_label).to(torch.float)

    val_label_path = os.path.join(filedir_in, 'val_label.pt')
    y_val_label = torch.load(val_label_path)
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
                                        dic=dic)
    df_val = pd.read_csv(val_path, sep='\t')
    x_val = fe.make_feature_pipeline(df=df_val['title'],
                                        dic=dic)

    batch_size = x_train.shape[1]
    batch_size_val = x_val.shape[1]

    net = RNN(input_size=input_size, 
            hidden_size=hidden_size,
            output_size=output_size,
            n_vocab=len(dic))

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(),
                    lr=0.001,
                    momentum=0.9)

    for epoch in range(config['epoch']):
        optimizer.zero_grad()

        output, h_T = net.forward(x=x_train, h_0=torch.zeros(1, batch_size, hidden_size))
        y_pred = output[-1, :, :]
        loss = criterion(y_pred, y_tr_label)
        tr_loss = loss.item()
        tr_acc = calc_acc(y_pred, y_tr_label)
        loss.backward()
        optimizer.step()

        output, h_T = net.forward(x=x_val, h_0=torch.zeros(1, batch_size_val, hidden_size))
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

if __name__ == '__main__':
    config = {
        'epoch': 10,
    }
    train(config=config)
    '''
    epoch: 1, tr_loss: 0.5939, tr_acc: 0.3961, val_loss: 0.5894, val_acc: 0.4003
    epoch: 2, tr_loss: 0.5929, tr_acc: 0.3961, val_loss: 0.5875, val_acc: 0.4003
    epoch: 3, tr_loss: 0.5910, tr_acc: 0.3961, val_loss: 0.5850, val_acc: 0.4003
    epoch: 4, tr_loss: 0.5883, tr_acc: 0.3961, val_loss: 0.5818, val_acc: 0.4010
    epoch: 5, tr_loss: 0.5850, tr_acc: 0.3961, val_loss: 0.5782, val_acc: 0.4010
    epoch: 6, tr_loss: 0.5812, tr_acc: 0.3961, val_loss: 0.5741, val_acc: 0.4010
    epoch: 7, tr_loss: 0.5769, tr_acc: 0.3961, val_loss: 0.5698, val_acc: 0.4010
    epoch: 8, tr_loss: 0.5723, tr_acc: 0.3961, val_loss: 0.5653, val_acc: 0.4010
    epoch: 9, tr_loss: 0.5675, tr_acc: 0.3961, val_loss: 0.5607, val_acc: 0.4010
    epoch: 10, tr_loss: 0.5626, tr_acc: 0.3961, val_loss: 0.5561, val_acc: 0.4010
    '''