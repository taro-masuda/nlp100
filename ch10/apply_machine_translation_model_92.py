'''
import onmt
import os
import random
import numpy as np

import yaml
import torch
import torch.nn as nn
from argparse import Namespace
from collections import defaultdict, Counter

from onmt.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields, IterOnDevice
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.translate import GNMTGlobalScorer, Translator, TranslationBuilder
from onmt.utils.misc import set_random_seed

from onmt.utils.logging import init_logger, logger
init_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    src_val = 'data/kftt-data-1.0/data/tok/kyoto-test.en'
    tgt_val = 'data/kftt-data-1.0/data/tok/kyoto-test.ja'
    model_path = 'data/run/en_ja_model_step_1000.pt'

    # initialize the frequency counter
    counters = defaultdict(Counter)

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
    fields = get_fields(
        'text', src_nfeats, tgt_nfeats)

    # build fields vocab
    share_vocab = False
    vocab_size_multiple = 1
    src_vocab_size = 30000
    tgt_vocab_size = 30000
    src_words_min_frequency = 1
    tgt_words_min_frequency = 1
    vocab_fields = _build_fields_vocab(
        fields, counters, 'text', share_vocab,
        vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency)

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab

    src_data = {"reader": onmt.inputters.str2reader["text"](), "data": src_val}
    tgt_data = {"reader": onmt.inputters.str2reader["text"](), "data": tgt_val}
    _readers, _data = onmt.inputters.Dataset.config(
        [('src', src_data), ('tgt', tgt_data)])

    dataset = onmt.inputters.Dataset(
        vocab_fields, readers=_readers, data=_data,
        sort_key=onmt.inputters.str2sortkey['text']
    )

    data_iter = onmt.inputters.OrderedIterator(
        dataset=dataset,
        device='cuda',
        batch_size=10,
        train=False,
        sort=False,
        sort_within_batch=True,
        shuffle=False
    )

    src_reader = onmt.inputters.str2reader['text']
    tgt_reader = onmt.inputters.str2reader['text']
    scorer = GNMTGlobalScorer(alpha=0.7,
                            beta=0.,
                            length_penalty='avg',
                            coverage_penalty='none')
    gpu = 0 if torch.cuda.is_available() else -1

    model = torch.load(model_path)#, map_location=torch.device('cpu'))
    translator = Translator(model=model,
                            fields=vocab_fields,
                            src_reader=src_reader,
                            tgt_reader=tgt_reader,
                            global_scorer=scorer,
                            gpu=gpu)
    builder = onmt.translate.TranslationBuilder(data=dataset,
                                                fields=vocab_fields)
    
    for batch in data_iter:
        trans_batch = translator.translate_batch(
            batch=batch, src_vocabs=[src_vocab],
            attn_debug=False)
        translations = builder.from_batch(trans_batch)
        for trans in translations:
            print(trans.log(0))
        break
'''