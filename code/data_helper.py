
from typing import *
import json
from constant import rel_dct, rel2id

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class Dependency():
    def __init__(self, idx, word, head, rel):
        self.id = idx
        self.word = word
        self.head = head
        self.rel = rel

    def __str__(self):
        # example:  1	上海	_	NR	NR	_	2	nn	_	_
        values = [str(self.idx), self.word, "_", "_", "_", "_", str(self.head), self.rel, "_", "_"]
        return '\t'.join(values)

    def __repr__(self):
        return f"({self.word}, {self.head}, {self.rel})"

    
def load_codt(data_file: str):
    sentence:List[Dependency] = []

    with open(data_file, 'r', encoding='utf-8') as f:
        # data example: 1	上海	_	NR	NR	_	2	nn	_	_
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:
                dep = Dependency(toks[0], toks[1], toks[3], toks[6], toks[7])
                sentence.append(dep)

def load_codt_new(data_file: str):
    sentence:List[Dependency] = []

    with open(data_file, 'r', encoding='utf-8') as f:
        # data example: 1	上海	_	NR	NR	_	2	nn	_	_
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:
                dep = Dependency(toks[0], toks[1], toks[3], toks[8], toks[9])
                sentence.append(dep)
                
def load_annoted(data_file, data_ids=None):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sample_lst:List[List[Dependency]] = []
    
    for i, d in enumerate(data):
        if data_ids is not None and i not in data_ids:
            continue
        rel_dct = {}
        for tripple in d['relationship']:
            head, rel, tail = tripple
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            if head_uttr_idx != tail_uttr_idx:
                continue
            
            if not rel_dct.get(head_uttr_idx, None):
                rel_dct[head_uttr_idx] = {tail_word_idx: [head_word_idx, rel]}
            else:
                rel_dct[head_uttr_idx][tail_word_idx] = [head_word_idx, rel]
            
        for item in d['dialog']:
            turn = item['turn']
            utterance = item['utterance']
            # dep_lst:List[Dependency] = [Dependency(0, '[root]', -1, '_')]
            dep_lst:List[Dependency] = []
            
            for word_idx, word in enumerate(utterance.split(' ')):
                head_word_idx, rel = rel_dct[turn].get(word_idx + 1, [word_idx, 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                dep_lst.append(Dependency(word_idx + 1, word, head_word_idx, rel))  # start from 1
            
            sample_lst.append(dep_lst)
        
    return sample_lst


class DialogDataset(Dataset):
    def __init__(self, cfg, data_file, data_ids):
        self.cfg = cfg
        self.data_file = data_file
        self.inputs, self.offsets, self.heads, self.rels, self.masks = self.read_data(data_ids)
        
    def read_data(self, data_ids):
        inputs, offsets = [], []
        tags, heads, rels, masks = [], [], [], []
        
        for deps in tqdm(load_annoted(self.data_file, data_ids)):
            # another sentence
            seq_len = len(deps)

            word_lst = [] 
            head_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)  # same as root index is 0, constrainting by mask 
            rel_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
            mask_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== self.cfg.max_length:
                    break

                word_lst.append(dep.word)

                if dep.head == -1 or dep.head + 1 >= self.cfg.max_length:
                    head_tokens[i+1] = 0
                    mask_tokens[i+1] = 0
                else:
                    head_tokens[i+1] = int(dep.head)
                    mask_tokens[i+1] = 1
                rel_tokens[i+1] = rel2id.get(dep.rel, 0)

            tokenized = self.cfg.tokenizer.encode_plus(word_lst, 
                                              padding='max_length', 
                                              truncation=True,
                                              max_length=self.cfg.max_length, 
                                              return_offsets_mapping=True, 
                                              return_tensors='pt',
                                              is_split_into_words=True)
            inputs.append({"input_ids": tokenized['input_ids'][0],
                          "token_type_ids": tokenized['token_type_ids'][0],
                           "attention_mask": tokenized['attention_mask'][0]
                          })

            sentence_word_idx = []
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)
            if len(sentence_word_idx) < self.cfg.max_length - 1:
                sentence_word_idx.extend([0]* (self.cfg.max_length - 1 - len(sentence_word_idx)))
            offsets.append(torch.as_tensor(sentence_word_idx))

            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
                    
        return inputs, offsets, heads, rels, masks

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx]
    
    def __len__(self):
        return len(self.rels)


class ConllDataset(Dataset):
    def __init__(self, cfg, load_fn, train):
        self.train = train
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer
        self.load_fn = load_fn
        self.inputs, self.offsets, self.heads, self.rels, self.masks = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        tags, heads, rels, masks = [], [], [], []
    
        file = self.cfg.train_file if self.train else self.cfg.dev_file
        for deps in tqdm(self.load_fn(file, self.train)):
            seq_len = len(deps)
            
            word_lst = [] 
            rel_attr = {'input_ids':torch.Tensor(), 'token_type_ids':torch.Tensor(), 'attention_mask':torch.Tensor()}
            head_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)  # same as root index is 0, constrainting by mask 
            rel_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
            mask_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== self.cfg.max_length:
                    break
                    
                word_lst.append(dep.word)
                    
                if dep.head in ['_', '-1'] or int(dep.head) + 1 >= self.cfg.max_length:
                    head_tokens[i+1] = 0
                    mask_tokens[i+1] = 0
                else:
                    head_tokens[i+1] = int(dep.head)
                    mask_tokens[i+1] = 1

                if self.train:
                    rel_tokens[i+1] = rel2id[dep.rel]
                else:
                    rel_tokens[i+1] = rel2id.get(dep.rel, rel2id['adjct'])

            tokenized = self.tokenizer.encode_plus(word_lst, 
                                                padding='max_length', 
                                                truncation=True,
                                                max_length=self.cfg.max_length, 
                                                return_offsets_mapping=True, 
                                                return_tensors='pt',
                                                is_split_into_words=True)
            inputs.append({"input_ids": tokenized['input_ids'][0],
                            "token_type_ids": tokenized['token_type_ids'][0],
                            "attention_mask": tokenized['attention_mask'][0]
                            })
            
            sentence_word_idx = []
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)
            if len(sentence_word_idx) < self.cfg.max_length - 1:
                sentence_word_idx.extend([0]* (self.cfg.max_length - 1 - len(sentence_word_idx)))
            offsets.append(torch.as_tensor(sentence_word_idx))
            
            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
                    
        return inputs, offsets, heads, rels, masks

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx]
    
    def __len__(self):
        return len(self.rels)


def load_codt_signal(data_file: str, return_two=False):
    sentence:List = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0 and len(sentence) != 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:                
                if return_two:
                    sentence.append([int(toks[2]), int(toks[3])])
                else:
                    sentence.append(int(toks[2]))


def load_inter(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    signal_iter = load_codt_signal('mlm_based/diag_test.conll', return_two=True)

    sample_lst:List[List[Dependency]] = []
    # for d, pred_signals in tqdm(zip(data, load_codt_signal('../prompt_based/diag_test.conll', idx=3))):
    for d in data:
        rel_dct = {}
        for tripple in d['relationship']:
            head, rel, tail = tripple
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            
            if rel == 'root' and head_uttr_idx != 0: # ignore root
                continue
                 
            if not rel_dct.get(tail_uttr_idx, None):
                rel_dct[tail_uttr_idx] = {tail_word_idx: [head, rel]}
            else:
                rel_dct[tail_uttr_idx][tail_word_idx] = [head, rel]
                
        sent_lens_accum = [1]
        for i, item in enumerate(d['dialog']):
            utterance = item['utterance']
            sent_lens_accum.append(sent_lens_accum[i] + len(utterance.split(' ')) + 1)
        sent_lens_accum[0] = 0
        
        dep_lst:List[Dependency] = []
        role_lst:List[str] = []
        weak_signal = []
        for item in d['dialog']:
            turn = item['turn']
            utterance = item['utterance']

            pred_signals = next(signal_iter)

            role = '[ans]' if item['speaker'] == 'A' else '[qst]'
            dep_lst.append(Dependency(sent_lens_accum[turn], role, -1, '_'))  
            
            tmp_signal = []
            for word_idx, word in enumerate(utterance.split(' ')):
                tail2head = rel_dct.get(turn, {1: [f'{turn}-{word_idx}', 'adjct']})
                head, rel = tail2head.get(word_idx + 1, [f'{turn}-{word_idx}', 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
                
                # only parse cross-utterance
                if turn != head_uttr_idx:
                    dep_lst.append(Dependency(sent_lens_accum[turn] + word_idx + 1, word, sent_lens_accum[head_uttr_idx] + head_word_idx, rel))  # add with accumulated length
                else:
                    dep_lst.append(Dependency(sent_lens_accum[turn] + word_idx + 1, word, -1, '_')) 

                try:
                    signal1, signal2 = pred_signals[i]
                except IndexError:
                    signal1, signal2 = pred_signals[len(pred_signals) - 1]
                
                tmp_signal = [signal1, signal2]
                
                # if word in weak_signal_dct.keys():
                #     tmp_signal.append(weak_signal_dct[word])

            if len(tmp_signal) != 0:
                # weak_signal.append(tmp_signal[-1])  # choose the last
                weak_signal.append(tmp_signal)  # choose the last
            else:
                weak_signal.append(-1)
            role_lst.append(item['speaker'])        
        sample_lst.append([dep_lst, role_lst, weak_signal])
        
    return sample_lst


class InterDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer= cfg.tokenizer
        self.inputs, self.offsets, self.heads, self.rels, self.masks, self.speakers, self.signs = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        tags, heads, rels, masks, speakers, signs = [], [], [], [], [], []
                
        for idx, (deps, roles, sign) in enumerate(load_inter(self.cfg.data_file)):
            seq_len = len(deps)
            signs.append(sign)

            word_lst = [] 
            head_tokens = np.zeros(1024, dtype=np.int64)  # same as root index is 0, constrainting by mask 
            rel_tokens = np.zeros(1024, dtype=np.int64)
            mask_tokens = np.zeros(1024, dtype=np.int64)
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== 1024:
                    break

                word_lst.append(dep.word)
                
                if int(dep.head) == -1 or int(dep.head) + 1 >= 1024:
                    head_tokens[i+1] = 0
                    mask_tokens[i+1] = 0
                else:
                    head_tokens[i+1] = int(dep.head)
                    mask_tokens[i+1] = 1
#                     head_tokens[i] = dep.head if dep.head != '_' else 0
                rel_tokens[i+1] = rel2id.get(dep.rel, 0)

            tokenized = self.tokenizer.encode_plus(word_lst, 
                                              padding='max_length', 
                                              truncation=True,
                                              max_length=1024, 
                                              return_offsets_mapping=True, 
                                              return_tensors='pt',
                                              is_split_into_words=True)
            inputs.append({"input_ids": tokenized['input_ids'][0],
                          "token_type_ids": tokenized['token_type_ids'][0],
                           "attention_mask": tokenized['attention_mask'][0]
                          })

#                 sentence_word_idx = np.zeros(self.cfg.max_length, dtype=np.int64)
            sentence_word_idx = []
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)
#                         sentence_word_idx[idx] = idx
            if len(sentence_word_idx) < 1024 - 1:
                sentence_word_idx.extend([0]* (1024 - 1 - len(sentence_word_idx)))
            offsets.append(torch.as_tensor(sentence_word_idx))
#                 offsets.append(sentence_word_idx)

            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
            speakers.append(roles)
                    
        return inputs, offsets, heads, rels, masks, speakers, signs

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx], self.speakers[idx], self.signs[idx]
    
    def __len__(self):
        return len(self.rels)