'''
import os
from janome.tokenizer import Token, Tokenizer
from transformers import AutoTokenizer
from torch import optim
from torch import nn
from torch.nn import (TransformerEncoder, TransformerDecoder,
                    TransformerEncoderLayer, TransformerDecoderLayer)
import random
import os
import torch
import numpy as np
from torch import Tensor
import math

def seed_everything(seed: int=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_dataset(filepath: str) -> list:
    texts = []
    with open(filepath, encoding='uft=8') as f:
        for line in f:
            tokens = line.strip().splot(' ')
            texts.append(tokens)
    return texts

class Preprocessing:
    def __init__(self):
        self.tokenizer = Tokenizer(wakati=False)
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    def preprocess_dataset(self, texts: list):
        return ['<start> {} <end>'.format(text) for text in texts]
    def preprocess_ja(self, texts: list):
        return [' '.join(self.tokenize(text)) for text in texts]

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
            self.pos_embedding[:token_embedding.size(0), :])

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int,
                num_decoder_layers: int,
                emb_size: int,
                src_vocab_size: int,
                tgt_vocab_size: int,
                dim_feedforward: int=512,
                dropout: float=0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size,
                                                nhead=NHEAD,
                                                dim_feedforward=dim_feedforward
                                                )
        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                    num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, 
                                                nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                                    num_layers=num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory, tgt_mask)

if __name__ == '__main__':
    seed_everything()

    in_dirpath = './data/kftt-data-1.0/data/tok/'
    en_train_texts = load_dataset(os.path.join(in_dirpath, 'kyoto-train.en'))
    jp_train_texts = load_dataset(os.path.join(in_dirpath, 'kyoto-train.ja'))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = 
    inputs = tokenizer.encode('translate English to German.')
    optimizer = optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
'''