
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

        mlp_arc_size:int = cfg.hidden_size
        mlp_rel_size:int = cfg.hidden_size
        
        self.cfg = cfg

        self.encoder = AutoModel.from_pretrained(cfg.plm)

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

        self.dropout = nn.Dropout(cfg.dropout)
     
    def feat(self, inputs):
        length = torch.sum(inputs["attention_mask"], dim=-1) - 2
        
        feats, *_ = self.encoder(**inputs, return_dict=False)   # batch_size, seq_len (tokenized), plm_hidden_size
           
        # remove [CLS] [SEP]
        word_cls = feats[:, :1]
        char_input = torch.narrow(feats, 1, 1, feats.size(1) - 2)
        return word_cls, char_input, length
        
    def forward(self, inputs, heads, offsets, evaluate=False):  # x: batch_size, seq_len
        cls_feat, char_feat, word_len = self.feat(inputs)
        word_idx = offsets.unsqueeze(-1).expand(-1, -1, char_feat.shape[-1])  # expand

        word_feat = torch.gather(char_feat, dim=1, index=word_idx)  # each char embedding of every word

        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        feats = self.dropout(word_cls_feat)
    
        all_dep = self.dropout(self.mlp_arc_dep(feats))  # batch_size, seq_len, hidden_size * 2
        all_head = self.dropout(self.mlp_arc_head(feats))

        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)

        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)  # batch_size, seq_len, hidden_size
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(arc_dep, arc_head)   # batch_size, seq_len, seq_len
        arc_logit = arc_logit.squeeze(3)

        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)  # batch_size, seq_len, hidden_size
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, rel_nums
        
        if evaluate:
            # change heads from golden to predicted
            _, heads = arc_logit.max(2)
            
        # expand: -1 means not changing the size of that dimension
        index = heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1]) 
        rel_logit = torch.gather(rel_logit_cond, dim=2, index=index).squeeze(2)  # batch_size, seq_len, rel_nums
        
        return feats, all_dep, arc_logit, rel_logit


class AttrEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = 'cuda' if cfg.cuda else 'cpu'

        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.dropout = nn.Dropout(cfg.dropout)

        self.split_idx = cfg.tokenizer('：')['input_ids'][1]
        
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, inputs):
        feats, *_ = self.encoder(**inputs, return_dict=False)  # batch_size, seq_len, hidden_size
        word_feat = torch.Tensor().to(self.device)
        split_ids = (inputs['input_ids'] == self.split_idx).nonzero()
        for idx, (i, j) in enumerate(split_ids):
            if i == split_ids[idx-1][0]:
                continue
            word_feat = torch.cat([word_feat, feats[i, 1:j].mean(0).unsqueeze(0)], dim=0)

        # return self.dropout(feats[:, 1, :])
        return word_feat


class ParWithAttr(nn.Module):
    def __init__(self, cfg, attr_tokenized):
        super().__init__()
        
        self.cfg = cfg
        self.device = 'cuda' if cfg.cuda else 'cpu'
        
        self.parser = DepParser(cfg)
        
        self.attr_encoder = AttrEncoder(cfg)
        
        self.head_hidden_size = cfg.hidden_size
        self.plm_hidden_size = self.attr_encoder.encoder.config.hidden_size

        # self.input_proj = nn.Linear(int(cfg.hidden_size * 2), self.plm_hidden_size)
        self.input_proj = nn.Linear(self.plm_hidden_size, self.plm_hidden_size)
        self.attr_proj = nn.Linear(self.plm_hidden_size, self.plm_hidden_size)

        self.dropout = nn.Dropout(cfg.dropout)
        
        self.init_attr_emb(attr_tokenized)
    
    def init_attr_emb(self, attr_tokenized):
        self.attr_encoder = self.attr_encoder.to(self.device)
        
        attr_tokenized_cuda = {}
        for key, value in attr_tokenized.items():
            attr_tokenized_cuda[key] = value.to(self.device)
        attr_tokenized = attr_tokenized_cuda
        
        attr_embeddings = self.attr_encoder(attr_tokenized)
        self.attr_embeddings = attr_embeddings.squeeze(1)  # num_rels, hidden_size
        # norm
        # self.attr_embeddings /= torch.norm(self.attr_embeddings, p=2, dim=1).unsqueeze(-1).expand(-1, self.plm_hidden_size)
    
    def forward(self, inputs, offsets, heads, rels, masks, evaluate=False):
        input_ctx_feat, input_rel_feat, arc_logit, rel_logit = self.parser(inputs, heads, offsets, evaluate)
        # batch_size, input_seq_len, head_hidden_size = input_feat.shape
        
        rels_onehot = F.one_hot(rels, 40)  # batch_size, input_seq_len, num_rels
        # attr_feat = torch.matmul(rels_onehot.float(), self.attr_embeddings)  # batch_size, input_seq_len, hidden_size
        
        # projection
        # input_feat = self.input_proj(input_rel_feat) - input_ctx_feat
        input_feat = self.dropout(torch.tanh(self.input_proj(torch.tanh(input_ctx_feat))))
        # attr_feat = self.attr_proj(self.attr_embeddings)
        attr_feat = self.dropout(torch.tanh(self.attr_proj(torch.tanh(self.dropout(self.attr_embeddings)))))
        
        groups = [[0], [1, 2, 3, *list(range(22, 35))], [4, 5], [6, 12, 13, 21], [7], [8, 9, 10], [11], [14], [15], [16], [17], [18], [19], [20], [35, 36, 37, 38, 39]]

        # norm
        input_feat_norm = input_feat / torch.norm(input_feat, p=2, dim=2).unsqueeze(-1).expand(-1, -1, self.plm_hidden_size)
        attr_feat_norm = attr_feat / torch.norm(attr_feat, p=2, dim=1).unsqueeze(-1).expand(-1, self.plm_hidden_size)
        
        # batch_size, input_seq_len, num_rels
        similarity = input_feat_norm @ attr_feat_norm.transpose(0, 1)
        inner_similarity = attr_feat_norm @ attr_feat_norm.transpose(0, 1) 
        # inner_similarity[torch.arange(40), torch.arange(40)] = 0

        if heads is None:
            return arc_logit, rel_logit
        else:
            loss_par = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)

            loss_inner = 0.0
            # for g in groups:
            #     if len(g) <= 1:
            #         continue
            #     idx = torch.tensor(g, dtype=torch.int64, device=self.device).unsqueeze(1).expand(-1, 40)
            #     tmp = inner_similarity.gather(dim=1, index=idx)
            #     tmp = tmp[:, 0]
            #     loss_inner += (1 - tmp).sum() / len(g)

            index = rels.unsqueeze(2).expand(-1, -1, similarity.size()[-1]) 
            pos = torch.gather(similarity, dim=2, index=index)  # batch_size, input_seq_len, num_rels
            pos = pos[:, :, 0]
            pos = 1 - pos
            # pos.masked_fill_((pos < 0), 0)

            for g in groups:
                if len(g) <= 1:
                    continue
                one_hot = F.one_hot(torch.tensor(g), 40)
                one_hot_sum = one_hot.sum(0).to(self.device)
                # print(one_hot_sum, one_hot_sum.shape)
                tmp = rels_onehot + one_hot_sum
                idx = (tmp != 0).sum(-1) > len(g)
                tmp.masked_fill_(idx.unsqueeze(-1).expand(-1, -1, 40).bool(), 0)

                rels_onehot += tmp
   
            neg, _ = similarity.masked_fill(rels_onehot.bool(), -1e4).max(-1)
            neg = neg - self.cfg.gamma
            neg.masked_fill_((neg < 0), 0)
            
            # loss_attr = self.cfg.gamma - pos + neg
            loss_attr = pos + neg
            # loss_attr.masked_fill_((loss_attr < 0), 0)
            loss_attr.masked_fill_(~masks.bool(), 0)
            loss_attr = loss_attr.sum(-1) / masks.sum(-1)
            loss_attr = loss_attr.sum()
            
            # flip_masks = masks.eq(0)  
            # loss_attr = F.cross_entropy(similarity.view(-1, similarity.size(-1)), rels.masked_fill(flip_masks, -1).view(-1), ignore_index=-1)
            loss = self.cfg.alpha * loss_par + (1 - self.cfg.alpha) * (self.cfg.beta * loss_attr + (1 - self.cfg.beta) * loss_inner)
            return arc_logit, rel_logit, loss

    def predict(self, inputs, offsets, masks):
        input_ctx_feat, input_rel_feat, arc_logit, rel_logit = self.parser(inputs, heads=None, offsets=offsets, evaluate=True)
        # batch_size, input_seq_len, head_hidden_size = input_feat.shape
        
        # rels_onehot = F.one_hot(rels, self.cfg.num_labels)  # batch_size, input_seq_len, num_labels
        # attr_feat = torch.matmul(rels_onehot.float(), self.attr_embeddings)  # batch_size, input_seq_len, hidden_size

        # projection
        input_feat = torch.tanh(self.input_proj(torch.tanh(input_ctx_feat)))
        # attr_feat = self.attr_proj(self.attr_embeddings)
        attr_feat = torch.tanh(self.attr_proj(torch.tanh(self.attr_embeddings)))
        
        # norm
        input_feat_norm = input_feat / torch.norm(input_feat, p=2, dim=2).unsqueeze(-1).expand(-1, -1, self.plm_hidden_size)
        attr_feat_norm = attr_feat / torch.norm(attr_feat, p=2, dim=1).unsqueeze(-1).expand(-1, self.plm_hidden_size)
        
        # batch_size, input_seq_len, num_rels
        similarity = input_feat_norm @ attr_feat_norm.transpose(0, 1)
        # similarity.masked_fill_(~masks.unsqueeze(-1).expand(-1, -1, attr_feat_norm.size(0)).bool(), -1e4)

        # batch_size, input_seq_len, num_rels
        # similarity = (input_feat / torch.norm(input_feat, p=2, dim=2).unsqueeze(-1).expand(-1, -1, self.plm_hidden_size)) @ self.attr_embeddings.transpose(0, 1)
        return arc_logit, rel_logit, similarity