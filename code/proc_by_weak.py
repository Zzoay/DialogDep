
import torch
from tqdm import tqdm

from data_helper import load_codt
from constant import rel2id, punct_lst, weak_signals, weak_labels


class CFG:
    # data_file = 'suda/train/BC-Train-full.conll'
    data_file = 'aug/diag_ensemble_logit/diag_train.conll'
    output_file = 'aug/diag_ensemble_logit/diag_train_weak.conll'


priority = {
    rel2id['cont']: 1,
    rel2id['temp']: 2, rel2id['cause']: 2, rel2id['bckg']: 2, rel2id['comp']: 2,
    rel2id['joint']: 3, rel2id['attr']: 3,
    None: 5,  # 4 is default
}

reverse_by_words = [
    '你 好', '您 好',
]

id2rel = list(rel2id.keys())

origin4change = [rel2id[x] for x in ['root', 'dfsubj', 'sasubj', 'elbr']]

signal_dct = {}
for i, signals in enumerate(weak_signals):
    for s in signals:
        signal_dct[s] = weak_labels[i]
print(signal_dct)

max_len = 160

f = open(CFG.output_file, 'w+', encoding='utf-8')

signals_whole = torch.Tensor()
heads_whole, rels_whole, masks_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
for deps in tqdm(load_codt(CFG.data_file)):
    seq_len = len(deps)
    if seq_len == 0:
        continue
    
    signals = torch.zeros(max_len).int()
    heads, masks, rels = torch.full(size=(max_len,), fill_value=-2).int(), torch.zeros(max_len).int(), torch.zeros(max_len).int()
    split, splits, signal, word_lst  = 1, [], None, ['root']
    word = deps[-1]
    for i, dep in enumerate(deps[:-1]):
        if i + 2 >= max_len:
            break

        word = dep.word
        word_lst.append(word)
        if word in signal_dct.keys() and priority.get(signal_dct[word], 4) < priority.get(signal, 4):
            signal = signal_dct[word]
        if f'{word} {deps[i+1].word}' in signal_dct.keys() and priority.get(signal_dct[f'{word} {deps[i+1].word}'], 4) < priority.get(signal, 4):
            signal = signal_dct[f'{word} {deps[i+1].word}']
        
        if word in punct_lst and deps[i+1].word not in punct_lst:
            if signal is not None and i + 2 - split > 2:  # set 2 to the min length of edu
                signals[split:i+2] = signal
                signal = None
            splits.append(split)
            split = i + 2

        if dep.head in ['_', '-1'] or int(dep.head) + 1 >= max_len:
            heads[i+1] = -1
            masks[i+1] = 0
        else:
            heads[i+1] = int(dep.head)
            masks[i+1] = 1
        rels[i+1] = rel2id.get(dep.rel, -1)

    # add the last data
    if i + 2 < max_len:
        word_lst.append(word)
        if deps[-1].head in ['_', '-1'] or int(deps[-1].head) + 1 >= max_len:
            heads[i+2] = -1
            masks[i+2] = 0
        else:
            heads[i+2] = int(deps[-1].head)
            masks[i+2] = 1
        rels[i+2] = rel2id.get(deps[-1].rel, -1)

    if split > 1:
        splits.append(split)
        if signal is not None:
            signals[split:i+2] = signal
    
    splits.append(len(deps))
    # when num of 'edu' >= 2, try change rel and head

    cnt = -1
    for idx, head in enumerate(heads[1:]):
        if head == -2:
            break
        if head == -1:
            continue

        if len(splits) > 2 and idx + 1 >= splits[cnt+1]:
            cnt += 1

        # if (head < splits[cnt] or head >= splits[cnt+1]) and rels[idx + 1] in origin4change:  # cross 'edu'
        if ((len(splits) > 2 and (head < splits[cnt] or head >= splits[cnt+1])) or idx - head > 2 or abs(idx - head) > 7) and rels[idx + 1] in origin4change:  # cross 'edu'
            if signals[idx+1] != 0:
                rels[idx+1] = signals[idx+1]
                
                if rels[idx + 1] in [rel2id['cond']]:  # reverse
                    tmp_heads = heads.clone()
                    tmp_heads[:splits[cnt+1]] = 0
                    head_idx = [idx + 1]
                    tail_idx = (tmp_heads == idx + 1).nonzero()  # find tail index
                    if len(tail_idx) == 0:  # ring or fail
                        tail_idx = [idx + 1]
                        head_idx = (heads == idx + 1).nonzero() if head_idx == tail_idx else head_idx
                    if len(head_idx) != 0:
                        heads[tail_idx[0]] = 0
                        heads[head_idx[0]] = tail_idx[0]
            elif head != 0:  # default
                rels[idx + 1] = rel2id['elbr']

            # special cases
            if len(splits) > 2 and word_lst[idx+1] == '好' and word_lst[idx] in ['你', '您']:  # reverse
                tmp_heads = heads.clone()
                tmp_heads[:splits[cnt+1]] = 0
                tail_idx = (tmp_heads == idx + 1).nonzero()  # find tail index
                if len(tail_idx) != 0:  
                    heads[tail_idx[0]] = 0
                    heads[idx + 1] = tail_idx[0]
                    rels[idx + 1] = rel2id['elbr']

            # 'attr' label should be reversed again; below can match most of cases
            if splits[cnt] == 1 and rels[idx + 1] in [rel2id['attr']]:
                tmp_heads = heads.clone()
                tmp_heads[:splits[cnt+1]] = 0
                tail_idx = ((tmp_heads != 0) * (tmp_heads >= splits[cnt]) * (tmp_heads < splits[cnt + 1])).nonzero().flatten()
                if len(tail_idx) != 0 and rels[tail_idx[0]] in origin4change:
                    heads[tail_idx[0]] = idx + 1
                    rels[tail_idx[0]] = rels[idx + 1]
                else:
                    dep_idx = heads[idx + 1].item()
                    if dep_idx != 0 and (dep_idx < splits[cnt] or dep_idx > splits[cnt+1]):
                        heads[idx + 1] = 0
                        heads[dep_idx] = idx + 1
                        rels[dep_idx] = rels[idx + 1]

    rels.masked_fill_(heads == 0, 0)  # root
    heads[0] = 0
    heads[1:].masked_fill_(heads[1:] == -2, 0)

    seq_len = seq_len if seq_len + 1 < max_len else max_len - 1
    for i in range(seq_len):
        new_head = heads[i+1].int().item()
        new_rel = id2rel[rels[i+1].int().item()]

        output_str = f'{deps[i].id}\t{deps[i].word}\t_\t{deps[i].tag}\t{deps[i].tag}\t_\t{deps[i].head}\t{deps[i].rel}\t{new_head}\t{new_rel}\n'
        f.write(output_str)
    f.write('\n')

    # heads_whole = torch.cat([heads_whole, heads.unsqueeze(0)])
    # rels_whole = torch.cat([rels_whole, rels.unsqueeze(0)])
    # masks_whole = torch.cat([masks_whole, masks.unsqueeze(0)])
    # signals_whole = torch.cat([signals_whole, signals.unsqueeze(0)])
f.close()


# for r in range(21, 40):
#     print((rels_whole == r).sum())


# # output
# f = open('aug/diag_train_weak.conll', 'w+', encoding='utf-8')
# for i, deps in enumerate(tqdm(load_codt(CFG.data_file))):
#     assert len(deps) > 0
#     for j, dep in enumerate(deps):
#         new_head = heads_whole[i][j+1].int().item()
#         new_rel = id2rel[rels_whole[i][j+1].int().item()]
        
#         output_str = f'{dep.id}\t{dep.word}\t_\t{dep.tag}\t{dep.tag}\t_\t{dep.head}\t{dep.rel}\t{new_head}\t{new_rel}\n'
#         f.write(output_str)
#     f.write('\n')

# f.close()