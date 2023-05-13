
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from model.module import NonLinear, Biaffine
from utils import arc_rel_loss


class DepParser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        mlp_arc_size:int = cfg.hidden_size
        mlp_rel_size:int = cfg.hidden_size
        dropout = cfg.dropout

        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.encoder.resize_token_embeddings(len(cfg.tokenizer))

        self.mlp_arc_dep = NonLinear(in_features=self.encoder.config.hidden_size, 
                                     out_features=mlp_arc_size+mlp_rel_size, 
                                     activation=nn.LeakyReLU(0.1))

        self.mlp_arc_head = NonLinear(in_features=self.encoder.config.hidden_size, 
                                      out_features=mlp_arc_size+mlp_rel_size, 
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((mlp_arc_size + mlp_rel_size) / 100)
        self.arc_num = int(mlp_arc_size / 100)
        self.rel_num = int(mlp_rel_size / 100)
        
        self.arc_biaffine = Biaffine(mlp_arc_size, mlp_arc_size, 1)
        self.rel_biaffine = Biaffine(mlp_rel_size, mlp_rel_size, cfg.num_labels)

        self.dropout = nn.Dropout(dropout)
     
    def feat(self, inputs):
        length = torch.sum(inputs["attention_mask"], dim=-1) - 2
        
        feats, *_ = self.encoder(**inputs, return_dict=False)   # batch_size, seq_len (tokenized), plm_hidden_size
           
        # remove [CLS] [SEP]
        word_cls = feats[:, :1]
        char_input = torch.narrow(feats, 1, 1, feats.size(1) - 2)
        return word_cls, char_input, length
        
    def forward(self, inputs, offsets, heads, rels, masks, evaluate=False):  # inputs: batch_size, seq_len
        cls_feat, char_feat, word_len = self.feat(inputs)
        
        word_idx = offsets.unsqueeze(-1).expand(-1, -1, char_feat.shape[-1])  # expand to the size of char feat
        word_feat = torch.gather(char_feat, dim=1, index=word_idx)  # embeddings of first char in each word

        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        feats = self.dropout(word_cls_feat)
    
        all_dep = self.dropout(self.mlp_arc_dep(feats))
        all_head = self.dropout(self.mlp_arc_head(feats))

        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)

        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(arc_dep, arc_head)   # batch_size, seq_len, seq_len
        arc_logit = arc_logit.squeeze(3)

        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels
        
        if evaluate:
            _, heads = arc_logit.max(2)  # change golden heads to the predicted
            
        index = heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1]) 
        rel_logit = torch.gather(rel_logit_cond, dim=2, index=index).squeeze(2)
        
        if evaluate:
            return arc_logit, rel_logit
        else:
            loss = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)
            return arc_logit, rel_logit, loss