
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel


class NonLinear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 activation: Optional[Callable] = None, 
                 init_func: Optional[Callable] = None) -> None: 
        super(NonLinear, self).__init__()
        self._linear = nn.Linear(in_features, out_features)
        self._activation = activation

        self.reset_parameters(init_func=init_func)
    
    def reset_parameters(self, init_func: Optional[Callable] = None) -> None:
        if init_func:
            init_func(self._linear.weight)

    def forward(self, x):
        if self._activation:
            return self._activation(self._linear(x))
        return self._linear(x)


class Biaffine(nn.Module):
    def __init__(self, 
                 in1_features: int, 
                 in2_features: int, 
                 out_features: int,
                 init_func: Optional[Callable] = None) -> None:
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.linear_in_features = in1_features 
        self.linear_out_features = out_features * in2_features

        # with bias default
        self._linear = nn.Linear(in_features=self.linear_in_features,
                                out_features=self.linear_out_features)

        self.reset_parameters(init_func=init_func)

    def reset_parameters(self, init_func: Optional[Callable] = None) -> None:
        if init_func:
            init_func(self._linear.weight)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        affine = self._linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine

    
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
        
    def forward(self, inputs, heads, offsets, evaluate=False):  # inputs: batch_size, seq_len
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
        
        return arc_logit, rel_logit
    

class DepParserTwostage(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = 'cuda' if self.cfg.cuda else 'cpu'

        self.mlp_arc_size = cfg.hidden_size
        self.mlp_rel_size = cfg.hidden_size

        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.encoder.resize_token_embeddings(len(cfg.tokenizer))

        self.mlp_arc_dep = NonLinear(in_features=self.encoder.config.hidden_size, 
                                     out_features=self.mlp_arc_size + self.mlp_rel_size, 
                                     activation=nn.LeakyReLU(0.1))

        self.mlp_arc_head = NonLinear(in_features=self.encoder.config.hidden_size, 
                                      out_features=self.mlp_arc_size + self.mlp_rel_size, 
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((self.mlp_arc_size + self.mlp_rel_size) / 100)
        self.arc_num = int(self.mlp_arc_size / 100)
        self.rel_num = int(self.mlp_rel_size / 100)
        
        self.arc_biaffine_inner = Biaffine(self.mlp_arc_size, self.mlp_arc_size, 1)
        self.rel_biaffine_inner = Biaffine(self.mlp_arc_size, self.mlp_rel_size, cfg.num_labels)
        
        self.arc_biaffine_mutual = Biaffine(self.mlp_arc_size, self.mlp_arc_size, 1)
        self.rel_biaffine_mutual = Biaffine(self.mlp_arc_size, self.mlp_rel_size, cfg.num_labels)

        self.dropout = nn.Dropout(cfg.dropout)
     
    def feat(self, inputs, offsets):
        length = torch.sum(inputs["attention_mask"], dim=-1) - 2
        
        feats, *_ = self.encoder(**inputs, return_dict=False)   # batch_size, seq_len (tokenized), plm_hidden_size
           
        # remove [CLS] [SEP]
        word_cls = feats[:, :1]
        char_input = torch.narrow(feats, 1, 1, feats.size(1) - 2)
        return word_cls, char_input, length
    
    def feat_word_level(self, cls_feat, char_feat, word_len, offsets):
        word_idx = offsets.unsqueeze(-1).expand(-1, -1, char_feat.shape[-1])  # expand to the size of char feat
        word_feat = torch.gather(char_feat, dim=1, index=word_idx)  # embeddings of first char in each word

        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        feats = self.dropout(word_cls_feat)
        all_dep = self.dropout(self.mlp_arc_dep(feats))
        all_head = self.dropout(self.mlp_arc_head(feats))
        return all_dep, all_head
    
    def parse_inner(self, all_dep, all_head, heads, evaluate=False):  # inputs: batch_size, seq_len 
        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)

        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine_inner(arc_dep, arc_head).squeeze(3)   # batch_size, seq_len, seq_len

        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine_inner(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels
        
        if evaluate:
            _, heads = arc_logit.max(2)  # change golden heads to the predicted
            
        index = heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1]) 
        rel_logit = torch.gather(rel_logit_cond, dim=2, index=index).squeeze(2)
        
        return arc_logit, rel_logit
    
    def parse_mutual(self, all_dep, all_head, heads, evaluate=False):  # inputs: batch_size, seq_len 
        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)

        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine_mutual(arc_dep, arc_head).squeeze(3)   # batch_size, seq_len, seq_len

        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine_mutual(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels
        
        if evaluate:
            _, heads = arc_logit.max(2)  # change golden heads to the predicted
            
        index = heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1]) 
        rel_logit = torch.gather(rel_logit_cond, dim=2, index=index).squeeze(2)
        
        return arc_logit, rel_logit
    
    def forward(self, inputs, heads, offsets, evaluate=False, predict=False):  # input_ids in inputs: batch_size, num_uttrs, seq_len
        arc_logit_inner, rel_logit_inner = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
        all_dep_whole, all_head_whole = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
        
        for idx in range(self.cfg.max_turns):
            # input_ids in inputs: batch_size, seq_len 
            inputs_uttr = {key:value[:, idx, :] for key, value in inputs.items()}
            
            if inputs_uttr['input_ids'][:, 0].sum(0) != 0.0:
                offsets_uttr, heads_uttr = offsets[:, idx, :], heads[:, idx, :]   # (batch_size, seq_len); (batch_size, seq_len)

                cls_feat, char_feat, word_len = self.feat(inputs_uttr, offsets_uttr)
                all_dep, all_head = self.feat_word_level(cls_feat, char_feat, word_len, offsets_uttr)  # batch_size, seq_len, hidden_size
                arc_logit, rel_logit = self.parse_inner(all_dep, all_head, heads_uttr, evaluate=evaluate)
            else:
                # arc_logit, rel_logit = torch.zeros(sefl.cfg.batch_size, self.cfg.max_length).to(self.device), torch.zeros(self.cfg.batch_size, self.cfg.num_labels).to(self.device)
                arc_logit, rel_logit = torch.zeros_like(arc_logit), torch.zeros_like(rel_logit)
            
            all_dep_whole = torch.cat([all_dep_whole, all_dep], dim=1)  # when loop end: batch_size, seq_len * num_uttrs, hidden_size
            all_head_whole = torch.cat([all_head_whole, all_head], dim=1)  # when loop end: batch_size, seq_len * num_uttrs, hidden_size
            
            arc_logit_inner = torch.cat([arc_logit_inner, arc_logit], dim=1)  # when loop end: batch_size, seq_len * num_uttrs, seq_len
            rel_logit_inner = torch.cat([rel_logit_inner, rel_logit], dim=1)  # when loop end: batch_size, seq_len * num_uttrs, num_rels
        
        if evaluate:
            heads_whole = None
        else:
            heads_whole = heads.view(self.cfg.batch_size, -1)  # batch_size, seq_len * num_uttrs
        
        batch_size, dialog_len, uttr_len = arc_logit_inner.size()
        arc_logit_inner = torch.cat([arc_logit_inner, torch.zeros(batch_size, dialog_len, dialog_len - self.cfg.max_length).to(self.device)], dim=2)  # batch_size, seq_len * num_uttrs, seq_len * num_uttrs
        arc_logit_masks = (arc_logit_inner != 0.0)
        
        # (batch_size, seq_len * num_uttrs, seq_len * num_uttrs); (batch_size, seq_len * num_uttrs, num_rels)
        arc_logit_mutual, rel_logit_mutual = self.parse_mutual(all_dep_whole, all_head_whole, heads_whole, evaluate=evaluate)  
        
        syntax_rst_bound = 21
        rel_logit_inner[:, :, syntax_rst_bound:] = rel_logit_mutual[:, :, syntax_rst_bound:]  # batch_size, seq_len * num_uttrs, num_rels
        
        # arc_logit_mutual = arc_logit_mutual * arc_logit_masks  # mask the utterance inner relations
        arc_logit_inner.masked_fill_((arc_logit_inner == 0.0), value=1e-9)
        arc_logit_mutual.masked_fill_((arc_logit_inner != 0.0), value=1e-9)  # mask the utterance inner relations
        
        # for convenience when predicting
        if predict:
            return arc_logit_inner, arc_logit_mutual, rel_logit_inner
        
        arc_logit_inner = arc_logit_inner + arc_logit_mutual
        return arc_logit_inner, rel_logit_inner  # they represent the whole dialogue now
    