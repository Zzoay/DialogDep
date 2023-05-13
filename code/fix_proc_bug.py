
from tqdm import tqdm

from constant import punct_lst


data_file = 'aug/diag_ensemble_logit/diag_train_weak.conll'

output_file = 'aug/diag_ensemble_logit/diag_train_fixed.conll'

fr = open(data_file, 'r', encoding='utf-8')
fw = open(output_file, 'w+', encoding='utf-8')

tmp = ''
bug_cnt = 0
root_idx = -1
cnt, subj_root_idx, reverse_subj_root, punc_idx = 1, -1, False, -1
for line in tqdm(fr.readlines()):
    toks = line.split()

    if len(toks) == 0:
        cnt = 1
        root_idx = -1
        fw.write('\n')
        continue
    
    idx, head = int(toks[0]), int(toks[8])
    if tmp in punct_lst:
        if toks[9] in ['dfsubj', 'sasubj', 'obj'] and head < idx:
            toks[9] = 'elbr'
            bug_cnt += 1
        punc_idx = idx
    elif (toks[9] in ['dfsubj', 'sasubj'] and (head < punc_idx < idx or idx - head > 3)) or (toks[9] in ['obj'] and (head < punc_idx < idx or idx - head > 8)):
        toks[9] = 'elbr'
        bug_cnt += 1
    
    if toks[9] == 'root':
        if root_idx == -1:
            root_idx = idx
        else:
            toks[8] = str(root_idx)  # only one root
            toks[9] = 'elbr' if (head < punc_idx < idx or idx - head > 4) else toks[7]

    elif toks[9] == 'subj' and head - idx > 10:
        subj_root_idx = idx
        reverse_subj_root = True
        toks[8] = '0'
        toks[9] = 'root'
        bug_cnt += 1
    elif toks[9] == 'root' and reverse_subj_root:
        reverse_subj_root = False
        toks[8] = str(subj_root_idx)
        toks[9] = 'elbr'

    cnt += 1
    new_line = '\t'.join(toks) + '\n'
    fw.write(new_line)
    tmp = toks[1]

print(bug_cnt)

fr.close()
fw.close()