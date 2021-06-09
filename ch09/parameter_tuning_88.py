from cnn_with_sgd_87 import train

if __name__ == '__main__':
    config = {
        'epoch': 50,
        'batch_size': 2,
        'input_size': 100,
        'output_size': 4
    }
    train(config=config)
    '''
    epoch: 50, tr_loss: 0.1332, tr_acc: 0.9074, val_loss: 0.3006, val_acc: 0.7751
    Time per epoch:  11.304419360160828 [s]
    '''