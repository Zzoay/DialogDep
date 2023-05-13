
from typing import *
from itertools import chain
import logging
import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.base_par import DepParser
from data_helper import Dependency, ConllDataset
from trainer import BasicTrainer
from utils import arc_rel_loss, uas_las, seed_everything


class CFG:
    train_file = 'aug/diag_weakcodt_sampled/diag_train_merged.conll'
    plm = 'hfl/chinese-electra-180g-base-discriminator'
    random_seed = 42
    num_epochs = 15
    batch_size = 32
    plm_lr = 2e-5
    head_lr = 1e-4
    weight_decay = 0.01
    dropout = 0.2
    grad_clip = 2
    scheduler = 'linear'
    warmup_ratio = 0.1
    num_early_stop = 3
    max_length = 160
    hidden_size = 400
    num_labels = 35
    print_every_ratio = 0.5
    cuda = True
    fp16 = True


logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
print(logger)
print(type(logger))
fh = logging.FileHandler(filename=f"results/res.log", mode='w')
logger.addHandler(fh)

time_now = datetime.datetime.now().isoformat()
print(time_now)
logger.info(f'=-=-=-=-=-=-=-=-={time_now}=-=-=-=-=-=-=-=-=-=')

seed_everything(CFG.random_seed)

if torch.cuda.is_available() and CFG.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
CFG.tokenizer = tokenizer

def load_conll(data_file: str):
    sentence:List[Dependency] = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:
                dep = Dependency(toks[0], toks[1], toks[8], toks[9])
                sentence.append(dep)

def load_conll_with_aug(data_file: str):
    sentence:List[Dependency] = []
    f1 = open(data_file, 'r', encoding='utf-8')
    f2 = open('aug/codt/codt_train_fixed.conll', 'r', encoding='utf-8')
    for line in chain(f1.readlines(), f2.readlines()):
        toks = line.split()
        if len(toks) == 0:
            yield sentence
            sentence = []
        elif len(toks) == 10:
            if toks[8] != '_':
                dep = Dependency(toks[0], toks[1],  toks[8], toks[9])
            else:
                dep = Dependency(toks[0], toks[1], toks[6], toks[7])
            sentence.append(dep)

    f1.close()
    f2.close()

train_dataset = ConllDataset(CFG, load_fn=load_conll_with_aug)
train_iter = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)

model = DepParser(CFG)

trainer = BasicTrainer(model=model, 
                       trainset_size=len(train_dataset), 
                       loss_fn=arc_rel_loss, 
                       metrics_fn=uas_las, 
                       logger=logger, 
                       config=CFG)

best_res, best_state_dict = trainer.train(model=model, train_iter=train_iter, val_iter=None)

torch.save(best_state_dict, f"results/base_par_new.pt")