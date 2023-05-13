
from typing import *

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from model.base_par import DepParser
from data_helper import Dependency, DialogDataset, load_annoted, InterDataset, load_codt_signal
from utils import uas_las, to_cuda, seed_everything
from constant import rel2id, punct_lst, weak_signals, weak_labels


class CFG:
    plm = 'hfl/chinese-electra-180g-base-discriminator'
    data_file = 'data/test.json'
    random_seed = 42
    num_epochs = 15
    batch_size = 128
    lr = 2e-5
    weight_decay = 0.01
    dropout = 0.2
    grad_clip = 2
    scheduler = 'linear'
    warmup_ratio = 0.1
    num_early_stop = 3
    max_length = 160
    hidden_size = 400
    num_labels = 35
    gamma = 0.7
    alpha = 0.7
    print_every = 400
    eval_every = 800
    cuda = True
    fp16 = True

seed_everything(CFG.random_seed)

if torch.cuda.is_available() and CFG.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
CFG.tokenizer = tokenizer

test_dataset = DialogDataset(CFG, data_file=CFG.data_file, data_ids=list(range(800)))
test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size)

model = DepParser(CFG)
print(model.load_state_dict(torch.load('results/par_codt+diag_merged.pt')))
model = model.cuda()

arc_logits, rel_logits, similarities = torch.Tensor(), torch.Tensor(), torch.Tensor()
heads_whole, rels_whole, masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
for batch in test_dataloader:
    inputs, offsets, heads, rels, masks = batch

    inputs_cuda = {}
    for key, value in inputs.items():
        inputs_cuda[key] = value.cuda()
    inputs = inputs_cuda

    offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
    
    with torch.no_grad():
        model.eval()
        arc_logit, rel_logit = model(inputs, offsets, heads, rels, masks, evaluate=True)
        
    # arc_logit[:, torch.arange(arc_logit.size()[1]), torch.arange(arc_logit.size()[2])] = -1e4
    arc_logit[:, 0, 1:] = float('-inf')
    arc_logit.diagonal(0, 1, 2)[1:].fill_(float('-inf'))
    
    arc_logits = torch.cat([arc_logits, arc_logit.cpu()])
    rel_logits = torch.cat([rel_logits, rel_logit.cpu()])

    heads_whole = torch.cat([heads_whole, heads.cpu()])
    rels_whole = torch.cat([rels_whole, rels.cpu()])
    masks_whole = torch.cat([masks_whole, masks.cpu()])

origin4change = [rel2id[x] for x in ['root', 'dfsubj', 'sasubj']]

signal_dct = {}
for i, signals in enumerate(weak_signals):
    for s in signals:
        signal_dct[s] = weak_labels[i]
print(signal_dct)

rel_preds = rel_logits.argmax(-1)
head_preds = arc_logits.argmax(-1)

max_len = CFG.max_length
signals_new_whole = torch.Tensor()
heads_new_whole, rels_new_whole = torch.Tensor(), torch.Tensor()
for sample_idx, (deps, pred_signals) in tqdm(enumerate(zip(load_annoted(CFG.data_file), load_codt_signal('mlm_based/diag_test.conll')))):
    seq_len = len(deps)
    if seq_len == 0:
        continue
    
    signals = torch.full(size=(max_len,), fill_value=rel2id['elbr']).int()
    heads, rels = torch.full(size=(max_len,), fill_value=-2).int(), torch.zeros(max_len).int()
    split, splits, signal, word_lst  = 1, [1], rel2id['elbr'], ['root']
    for i, dep in enumerate(deps[:-1]):
        if i + 2 >= max_len:
            break
        
        word = dep.word
        word_lst.append(word)

        try:
            signal = pred_signals[i]
        except IndexError:
            signal = pred_signals[len(pred_signals) - 1]

        if word in punct_lst and deps[i+1].word not in punct_lst:
            if i + 2 - split > 2:  # set 2 to the min length of edu
                signals[split:i+2] = signal
            split = i + 2
            splits.append(split)

    splits.append(len(deps))
            
    # add the last data
    if i + 1 < max_len:
        signal = pred_signals[-1]
        word_lst.append(word)

    heads = head_preds[sample_idx]
    heads.masked_fill_(mask=~masks_whole[sample_idx].bool(), value=-2)

    rels = rel_preds[sample_idx]
    rels.masked_fill_(mask=~masks_whole[sample_idx].bool(), value=-2)

    cnt, attr, = -1, False
    for idx, head in enumerate(heads[1:]):
        if head == -2:
            break
        if head == -1:
            continue

        if len(splits) > 2 and idx + 1 >= splits[cnt+1]:
            cnt += 1

        if ((len(splits) > 2 and (head < splits[cnt] or head >= splits[cnt+1])) or idx - head > 0) and rels[idx + 1] in origin4change:  # cross 'edu'

            rels[idx+1] = signals[idx+1]

            if rels[idx + 1] in [rel2id['cond']] or (not attr and rels[idx + 1] == rel2id['attr']):  # reverse
                tmp_heads = heads.clone()
                tmp_heads[:splits[cnt+1]] = 0
                head_idx = [idx + 1]
                tail_idx = (tmp_heads == idx + 1).nonzero()  # find tail index
                if len(tail_idx) == 0:  # ring or fail
                    # unchange
                    tail_idx = [idx + 1]
                    head_idx = (heads == idx + 1).nonzero() if head_idx == tail_idx else head_idx
                elif len(head_idx) != 0:
                    heads[tail_idx[0]] = 0
                    heads[head_idx[0]] = tail_idx[0]

            # special cases
            if word_lst[idx+1] == '好' and word_lst[idx] in ['你', '您']:  # reverse
                tmp_heads = heads.clone()
                tmp_heads[:splits[cnt+1]] = 0
                tail_idx = (tmp_heads == idx + 1).nonzero()  # find tail index
                if len(tail_idx) != 0:  
                    heads[tail_idx[0]] = 0
                    heads[idx + 1] = tail_idx[0]
                    rels[idx + 1] = rel2id['elbr']

    rels.masked_fill_(heads == 0, 0)  # root
    heads[0] = 0
    heads[1:].masked_fill_(heads[1:] == -2, 0)

    heads_new_whole = torch.cat([heads_new_whole, heads.unsqueeze(0)])
    rels_new_whole = torch.cat([rels_new_whole, rels.unsqueeze(0)])
    signals_new_whole = torch.cat([signals_new_whole, signals.unsqueeze(0)])

arc_logits_correct = (heads_new_whole == heads_whole).long() * masks_whole * (rels_whole >= 21).long()
rel_logits_correct = (rels_new_whole == rels_whole).long() * arc_logits_correct
print('inner-utterance (RST):')
print(rel_logits_correct.sum() / (rels_whole >= 21).long().sum())
print(arc_logits_correct.sum() / (rels_whole >= 21).long().sum())
print('---------------------------------------------------')

arc_logits_correct = (heads_new_whole == heads_whole).long() * masks_whole * (rels_whole < 21).long()
rel_logits_correct = (rels_new_whole == rels_whole).long() * arc_logits_correct 
print('inner-EDU (syntax):')
print(rel_logits_correct.sum() / (masks_whole * (rels_whole < 21).long()).sum())
print(arc_logits_correct.sum() / (masks_whole * (rels_whole < 21).long()).sum())

root_ids = []
for rel_pred, mask in zip(rel_preds, masks_whole):
    try:
        root_idx = (((rel_pred == 0) * mask) != 0).nonzero()[0].item()
    except IndexError: # no root
        root_idx = 2
    root_ids.append(root_idx)
print(root_ids[:10])

inter_dataset = InterDataset(CFG)
inter_dataloader = DataLoader(inter_dataset, batch_size=1)

cnt = 0
inter_heads_whole, inter_rels_whole, inter_masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
inter_heads_preds, inter_rels_preds = torch.Tensor(), torch.Tensor()
for batch in inter_dataloader:
    inputs, offsets, heads, rels, masks, speakers, signs = batch
    inter_head_preds = torch.zeros_like(heads, dtype=int)
    inter_rel_preds = torch.zeros_like(rels, dtype=int)

    inter_heads_whole = torch.cat([inter_heads_whole, heads])
    inter_rels_whole = torch.cat([inter_rels_whole, rels])
    inter_masks_whole = torch.cat([inter_masks_whole, masks])

    accum = 1
    for i, speakr in enumerate(speakers[1:]):
        seq_len = masks_whole[cnt].sum().item() + 1
        if speakr == speakers[i]:
            rel = signs[i][0]
        else:
            rel = signs[i][1]
        
        head_idx = int(root_ids[cnt] + accum) if i > 0 else root_ids[cnt]
        tail_idx = int(root_ids[cnt+1] + accum + seq_len)
        
        inter_head_preds[0][tail_idx] = head_idx
        inter_rel_preds[0][tail_idx] = rel

        cnt += 1
        accum += seq_len

    cnt += 1

    inter_heads_preds = torch.cat([inter_heads_preds, inter_head_preds])
    inter_rels_preds = torch.cat([inter_rels_preds, inter_rel_preds])

arc_logits_correct_inner = (head_preds == heads_whole).long() * masks_whole * (rels_whole >= 21).long()
arc_logits_correct_inter = (inter_heads_preds == inter_heads_whole).long() * inter_masks_whole
inter_edu_uas = (arc_logits_correct_inner.sum() + arc_logits_correct_inter.sum()) / ((rels_whole >= 21).long().sum() + inter_masks_whole.long().sum())

rel_logits_correct_inner = (rel_preds == rels_whole).long() * arc_logits_correct_inner
rel_logits_correct_inter = (inter_rels_preds == inter_rels_whole).long() * arc_logits_correct_inter
inter_edu_las = (rel_logits_correct_inner.sum() + rel_logits_correct_inter.sum()) / ((rels_whole >= 21).long().sum() + inter_masks_whole.long().sum())

print('inter-EDU:')
print(inter_edu_uas)
print(inter_edu_las)

print('----------end----------')