
from collections import Counter
from typing import *
import random
import json
import logging
import datetime

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

import sys
sys.path.append('..')

from trainer import BasicTrainer
from model.base_par import DepParser
from data_helper import Dependency, load_annoted, DialogDataset
from utils import arc_rel_loss, uas_las, to_cuda, seed_everything
# from constant import rel_dct, rel2id

transformers.logging.set_verbosity_error() # only report errors.


class CFG:
    data_file = '../data/train_50.json'
    test_file = '../data/test.json'
    plm = 'hfl/chinese-electra-180g-base-discriminator'
    random_seeds = [40, 41, 42, 43, 44]
    shot = 10
    num_epochs = 30
    batch_size = 32
    plm_lr = 2e-5
    head_lr = 1e-4
    weight_decay = 0.01
    dropout = 0.2
    grad_clip = 1
    scheduler = 'linear'
    warmup_ratio = 0.1
    num_early_stop = 5
    max_length = 160
    num_labels = 35
    hidden_size = 400
    print_every = 3
    # eval_every = 100
    cuda = True
    fp16 = True
    eval_strategy = 'epoch'
    mode = 'training'
    

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
print(logger)
print(type(logger))

if CFG.mode == 'training':
    fh = logging.FileHandler(filename=f"results/few_shot/{CFG.shot}-shot/res.log",mode='w')
else:
    fh = logging.FileHandler(filename=f"results/few_shot/res.log",mode='w')
logger.addHandler(fh)

time_now = datetime.datetime.now().isoformat()
print(time_now)
logger.info(f'=-=-=-=-=-=-=-=-={time_now}=-=-=-=-=-=-=-=-=-=')

tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
print(len(tokenizer))
CFG.tokenizer = tokenizer

def evaluate(model, eval_iter):
    arc_logits, rel_logits = torch.Tensor(), torch.Tensor()
    heads_whole, rels_whole, masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
    for batch in eval_iter:
        inputs, offsets, heads, rels, masks = batch

        inputs_cuda = {}
        for key, value in inputs.items():
            inputs_cuda[key] = value.cuda()
        inputs = inputs_cuda

        offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))

        with torch.no_grad():
            model.eval()
            arc_logit, rel_logit = model(inputs, offsets, heads, rels, masks, evaluate=True)

        arc_logit[:, torch.arange(arc_logit.size()[1]), torch.arange(arc_logit.size()[2])] = -1e4

        arc_logits = torch.cat([arc_logits, arc_logit.cpu()])
        rel_logits = torch.cat([rel_logits, rel_logit.cpu()])

        heads_whole = torch.cat([heads_whole, heads.cpu()])
        rels_whole = torch.cat([rels_whole, rels.cpu()])
        masks_whole = torch.cat([masks_whole, masks.cpu()])

    rel_preds = rel_logits.argmax(-1)
    head_pred = arc_logits.argmax(-1)

    arc_logits_correct = (head_pred == heads_whole).long() * masks_whole * (rels_whole < 21).long()
    rel_logits_correct = (rel_preds == rels_whole).long() * arc_logits_correct 
    uas_syntax = (arc_logits_correct.sum() / (masks_whole * (rels_whole < 21).long()).sum()).item()
    las_syntax = (rel_logits_correct.sum() / (masks_whole * (rels_whole < 21).long()).sum()).item()
    logger.info(f'Syntax UAS: {uas_syntax}; Syntax LAS: {las_syntax}')

    arc_logits_correct = (head_pred == heads_whole).long() * masks_whole * (rels_whole >= 21).long()
    rel_logits_correct = (rel_preds == rels_whole).long() * arc_logits_correct
    uas_discourse = (arc_logits_correct.sum() / (rels_whole >= 21).long().sum()).item()
    las_discourse = (rel_logits_correct.sum() / (rels_whole >= 21).long().sum()).item()
    logger.info(f'Discourse UAS: {uas_discourse}; Discourse LAS: {las_discourse}')
    logger.info('---------------------------------------------------')
    
    return uas_syntax, las_syntax, uas_discourse, las_discourse


def run():
    if CFG.mode == 'training':    
        total_ids = list(range(50))
        for seed in CFG.random_seeds:   
            print(f'\nSEED {seed}')
            print('--------------------------------')
            logger.info(f'\n=========SEED {seed}===========')
            logger.info(f'-------------------------------')

            if CFG.cuda and torch.cuda.is_available:
                torch.cuda.empty_cache()

            seed_everything(seed=seed)

            random.shuffle(total_ids)

            train_ids = total_ids[0:CFG.shot]
            val_ids = total_ids[CFG.shot:2*CFG.shot]

            tr_dataset = DialogDataset(CFG, CFG.data_file, train_ids)
            va_dataset = DialogDataset(CFG, CFG.data_file, val_ids)

            print(f'---Data Size Train/Val: {len(tr_dataset)} / {len(va_dataset)}')
            logger.info(f'---Data Size Train/Val: {len(tr_dataset)} / {len(va_dataset)}')

            tr_iter = DataLoader(tr_dataset, batch_size=CFG.batch_size)
            va_iter = DataLoader(va_dataset, batch_size=CFG.batch_size * 2)

            model = DepParser(CFG)
            print('Loading Model....')
            trainer = BasicTrainer(model=model, 
                                   trainset_size=len(tr_dataset), 
                                   loss_fn=arc_rel_loss, 
                                   metrics_fn=uas_las, 
                                   logger=logger, 
                                   config=CFG)

            best_res, best_state_dict = trainer.train(model=model, train_iter=tr_iter, val_iter=va_iter)
            print(best_res)
            with open(f"results/few_shot/{CFG.shot}-shot/res.txt", 'a+') as f:
                f.write(f'{seed}\t {str(best_res)}\n')

            torch.save(best_state_dict, f'results/few_shot/{CFG.shot}-shot/model_{seed}.bin')

            model = None

        logger.info('\n')
        
    test_dataset = DialogDataset(CFG, CFG.test_file, list(range(800)))
    test_iter = DataLoader(test_dataset, batch_size=CFG.batch_size * 6)
    
    if CFG.mode == 'training':
        model = DepParser(CFG)
        uas_syntaxs, las_syntaxs = [], []
        uas_discourses, las_discourses = [], []
        for seed in CFG.random_seeds:
            model.load_state_dict(torch.load(f'results/few_shot/{CFG.shot}-shot/model_{seed}.bin'))
            model = model.cuda()

            uas_syntax, las_syntax, uas_discourse, las_discourse = evaluate(model, test_iter)

            uas_syntaxs.append(uas_syntax)
            las_syntaxs.append(las_syntax)
            uas_discourses.append(uas_discourse)
            las_discourses.append(las_discourse)

        avg_uas_syntax, avg_las_syntax = np.mean(uas_syntaxs), np.mean(las_syntaxs)
        avg_uas_discourse, avg_las_discourse = np.mean(uas_discourses), np.mean(las_discourses)
        std_uas_syntax, std_las_syntax = np.std(uas_syntaxs), np.std(las_syntaxs)
        std_uas_discourse, std_las_discourse = np.std(uas_discourses), np.std(las_discourses)
        logger.info('\n----------------Result----------------')
        logger.info(f'Avg Syntax UAS: {avg_uas_syntax:.4f}; Avg Syntax LAS: {avg_las_syntax:.4f}')
        logger.info(f'Std Syntax UAS: {std_uas_syntax:.4f}; Std Syntax LAS: {std_las_syntax:.4f}')
        logger.info(f'Avg Discourse UAS: {avg_uas_discourse:.4f}; Avg Discourse LAS: {avg_las_discourse:.4f}')
        logger.info(f'Std Discourse UAS: {std_uas_discourse:.4f}; Std Discourse LAS: {std_las_discourse:.4f}\n')

    if CFG.mode == 'inference':
        model = DepParser(CFG)

        for shot in [5, 10, 20, 50]:
            logger.info(f'----------------Shot: {shot}----------------')
            uas_syntaxs, las_syntaxs = [], []
            uas_discourses, las_discourses = [], []
            for seed in tqdm(CFG.random_seeds):
                model.load_state_dict(torch.load(f'results/few_shot/{shot}-shot/model_{seed}.bin'))
                model = model.cuda()

                uas_syntax, las_syntax, uas_discourse, las_discourse = evaluate(model, test_iter)

                uas_syntaxs.append(uas_syntax)
                las_syntaxs.append(las_syntax)
                uas_discourses.append(uas_discourse)
                las_discourses.append(las_discourse)

            avg_uas_syntax, avg_las_syntax = np.mean(uas_syntaxs), np.mean(las_syntaxs)
            avg_uas_discourse, avg_las_discourse = np.mean(uas_discourses), np.mean(las_discourses)
            std_uas_syntax, std_las_syntax = np.std(uas_syntaxs), np.std(las_syntaxs)
            std_uas_discourse, std_las_discourse = np.std(uas_discourses), np.std(las_discourses)
            logger.info('\n----------------Result----------------')
            logger.info(f'Avg Syntax UAS: {avg_uas_syntax:.4f}; Avg Syntax LAS: {avg_las_syntax:.4f}')
            logger.info(f'Std Syntax UAS: {std_uas_syntax:.4f}; Std Syntax LAS: {std_las_syntax:.4f}')
            logger.info(f'Avg Discourse UAS: {avg_uas_discourse:.4f}; Avg Discourse LAS: {avg_las_discourse:.4f}')
            logger.info(f'Std Discourse UAS: {std_uas_discourse:.4f}; Std Discourse LAS: {std_las_discourse:.4f}\n')

    logger.info('=================End=================')
    logger.info(datetime.datetime.now().isoformat())
    logger.info('=====================================')
        
if __name__ == '__main__':
    run()