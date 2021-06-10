import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch import optim
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from cnn_with_sgd_87 import WordEmbeddingExtractor, calc_acc
from rnn_prediction_81 import seed_everything
from sklearn import preprocessing
import os

class BERT(nn.Module):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                            return_dict=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.softmax = nn.Softmax(dim=1)
        self.critetion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.bert(input_ids, attention_mask=attention_mask)
        x = self.fc(x.pooler_output)
        output = self.softmax(x)
        loss = 0
        if labels is not None:
            loss = self.critetion(output, labels)
        return loss, output

class TextDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, 
                max_token_len: int=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.le = preprocessing.LabelEncoder()
        self.le.fit(data['category'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        title = data_row.title
        label = data_row['category']
        label = torch.tensor(self.le.transform([label]), dtype=torch.int64)
        labels = torch.nn.functional.one_hot(label, num_classes=len(self.le.classes_))
        labels = labels.squeeze().type(torch.FloatTensor)

        encoding = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return dict(
            comment_text=title,
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten(),
            labels=labels
        )

def train(config: dict) -> None:
    seed_everything()
    filedir_in = './data'

    if torch.cuda.is_available():
      device = torch.device('cuda:0') 
      torch.cuda.empty_cache()
    else:
      device = torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)    

    train_path = os.path.join(filedir_in, 'train.txt')
    val_path = os.path.join(filedir_in, 'val.txt')
    df_train = pd.read_csv(train_path, sep='\t')
    df_val = pd.read_csv(val_path, sep='\t')

    train_label_path = os.path.join(filedir_in, 'train_label.pt')
    y_tr_label = torch.load(train_label_path).to(device)
    y_tr_label = torch.nn.functional.one_hot(y_tr_label).to(torch.float)

    tr_dataset = TextDataset(tokenizer=tokenizer,
                        data=df_train,
                        max_token_len=config['max_seq_len'])
    train_loader = DataLoader(tr_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataset = TextDataset(tokenizer=tokenizer,
                        data=df_val,
                        max_token_len=config['max_seq_len'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    total_training_steps = config['epoch'] * (len(df_train) // config['batch_size'])
    warmup_steps = total_training_steps // 5
    criterion = nn.BCELoss()
    model = BERT(
        n_classes=config['output_size'],
        n_training_steps=total_training_steps,
        n_warmup_steps=warmup_steps
    ).to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(config['epoch']):
        tr_loss, tr_acc = 0, 0
        for x_tr in train_loader:

            loss, output = model(x_tr['input_ids'].to(device), x_tr['attention_mask'].to(device), x_tr['labels'].to(device))
            
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            tr_acc += calc_acc(output, x_tr['labels'].to(device))
        tr_loss /= (len(df_train) // config['batch_size'])
        tr_acc /= (len(df_train) // config['batch_size'])

        val_loss, val_acc = 0, 0
        for x_val in val_loader:

            loss, output = model(x_val['input_ids'].to(device), x_val['attention_mask'].to(device), x_val['labels'].to(device))
            
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            val_loss += loss.item()
            val_acc += calc_acc(output, x_val['labels'].to(device))
        val_loss /= (len(df_val) // config['batch_size'])
        val_acc /= (len(df_val) // config['batch_size'])

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
        'epoch': 50,
        'batch_size': 16,
        'output_size': 4,
        'max_seq_len': 512
    }
    train(config=config)
    '''
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 465.27       Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   33C    P0    23W / 300W |      0MiB / 16160MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    epoch: 1, tr_loss: 0.5126, tr_acc: 0.4119, val_loss: 0.5179, val_acc: 0.4162
    epoch: 2, tr_loss: 0.5082, tr_acc: 0.4011, val_loss: 0.5289, val_acc: 0.4006
    epoch: 3, tr_loss: 0.5089, tr_acc: 0.4153, val_loss: 0.5186, val_acc: 0.4096
    epoch: 4, tr_loss: 0.5032, tr_acc: 0.4134, val_loss: 0.5252, val_acc: 0.4089
    epoch: 5, tr_loss: 0.5038, tr_acc: 0.4156, val_loss: 0.5335, val_acc: 0.3951
    epoch: 6, tr_loss: 0.5090, tr_acc: 0.4086, val_loss: 0.5121, val_acc: 0.3732
    epoch: 7, tr_loss: 0.5039, tr_acc: 0.4092, val_loss: 0.5158, val_acc: 0.4169
    epoch: 8, tr_loss: 0.4998, tr_acc: 0.4055, val_loss: 0.5141, val_acc: 0.3604
    epoch: 9, tr_loss: 0.4980, tr_acc: 0.4077, val_loss: 0.5070, val_acc: 0.4199
    epoch: 10, tr_loss: 0.4906, tr_acc: 0.4087, val_loss: 0.5037, val_acc: 0.4204
    epoch: 11, tr_loss: 0.4896, tr_acc: 0.4093, val_loss: 0.5022, val_acc: 0.4355
    epoch: 12, tr_loss: 0.4887, tr_acc: 0.4202, val_loss: 0.5032, val_acc: 0.4099
    epoch: 13, tr_loss: 0.4888, tr_acc: 0.4142, val_loss: 0.5028, val_acc: 0.3913
    epoch: 14, tr_loss: 0.4885, tr_acc: 0.4166, val_loss: 0.5036, val_acc: 0.3893
    epoch: 15, tr_loss: 0.4882, tr_acc: 0.4257, val_loss: 0.5017, val_acc: 0.4009
    epoch: 16, tr_loss: 0.4883, tr_acc: 0.4139, val_loss: 0.5036, val_acc: 0.4096
    epoch: 17, tr_loss: 0.4883, tr_acc: 0.4192, val_loss: 0.5029, val_acc: 0.3988
    epoch: 18, tr_loss: 0.4881, tr_acc: 0.4164, val_loss: 0.5032, val_acc: 0.4009
    epoch: 19, tr_loss: 0.4883, tr_acc: 0.4191, val_loss: 0.5026, val_acc: 0.4049
    epoch: 20, tr_loss: 0.4880, tr_acc: 0.4148, val_loss: 0.5021, val_acc: 0.4056
    epoch: 21, tr_loss: 0.4876, tr_acc: 0.4186, val_loss: 0.5016, val_acc: 0.4096
    epoch: 22, tr_loss: 0.4885, tr_acc: 0.4176, val_loss: 0.5011, val_acc: 0.4016
    epoch: 23, tr_loss: 0.4879, tr_acc: 0.4193, val_loss: 0.5018, val_acc: 0.3870
    epoch: 24, tr_loss: 0.4881, tr_acc: 0.4143, val_loss: 0.5018, val_acc: 0.4014
    epoch: 25, tr_loss: 0.4877, tr_acc: 0.4193, val_loss: 0.5020, val_acc: 0.4046
    epoch: 26, tr_loss: 0.4882, tr_acc: 0.4195, val_loss: 0.5022, val_acc: 0.3986
    epoch: 27, tr_loss: 0.4876, tr_acc: 0.4241, val_loss: 0.5014, val_acc: 0.4109
    epoch: 28, tr_loss: 0.4883, tr_acc: 0.4180, val_loss: 0.5030, val_acc: 0.4071
    epoch: 29, tr_loss: 0.4876, tr_acc: 0.4220, val_loss: 0.5020, val_acc: 0.4071
    epoch: 30, tr_loss: 0.4881, tr_acc: 0.4141, val_loss: 0.5011, val_acc: 0.4116
    epoch: 31, tr_loss: 0.4875, tr_acc: 0.4162, val_loss: 0.5028, val_acc: 0.3983
    epoch: 32, tr_loss: 0.4879, tr_acc: 0.4181, val_loss: 0.5034, val_acc: 0.4056
    epoch: 33, tr_loss: 0.4875, tr_acc: 0.4197, val_loss: 0.5034, val_acc: 0.3923
    epoch: 34, tr_loss: 0.4877, tr_acc: 0.4201, val_loss: 0.5013, val_acc: 0.4104
    epoch: 35, tr_loss: 0.4877, tr_acc: 0.4236, val_loss: 0.5021, val_acc: 0.4101
    epoch: 36, tr_loss: 0.4877, tr_acc: 0.4122, val_loss: 0.5022, val_acc: 0.4016
    epoch: 37, tr_loss: 0.4875, tr_acc: 0.4240, val_loss: 0.5028, val_acc: 0.3963
    epoch: 38, tr_loss: 0.4881, tr_acc: 0.4200, val_loss: 0.5023, val_acc: 0.4024
    epoch: 39, tr_loss: 0.4875, tr_acc: 0.4230, val_loss: 0.5017, val_acc: 0.4081
    epoch: 40, tr_loss: 0.4878, tr_acc: 0.4197, val_loss: 0.5016, val_acc: 0.3956
    epoch: 41, tr_loss: 0.4875, tr_acc: 0.4227, val_loss: 0.5014, val_acc: 0.4006
    epoch: 42, tr_loss: 0.4881, tr_acc: 0.4177, val_loss: 0.5018, val_acc: 0.4051
    epoch: 43, tr_loss: 0.4874, tr_acc: 0.4229, val_loss: 0.5029, val_acc: 0.4039
    epoch: 44, tr_loss: 0.4879, tr_acc: 0.4187, val_loss: 0.5014, val_acc: 0.4071
    epoch: 45, tr_loss: 0.4875, tr_acc: 0.4211, val_loss: 0.5029, val_acc: 0.3853
    epoch: 46, tr_loss: 0.4875, tr_acc: 0.4215, val_loss: 0.5013, val_acc: 0.3971
    epoch: 47, tr_loss: 0.4880, tr_acc: 0.4171, val_loss: 0.5028, val_acc: 0.4084
    epoch: 48, tr_loss: 0.4872, tr_acc: 0.4200, val_loss: 0.5010, val_acc: 0.4121
    epoch: 49, tr_loss: 0.4875, tr_acc: 0.4244, val_loss: 0.5024, val_acc: 0.3818
    epoch: 50, tr_loss: 0.4875, tr_acc: 0.4198, val_loss: 0.5018, val_acc: 0.4164
    '''