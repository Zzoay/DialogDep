
from typing import *

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import to_cuda


class BasicTrainer():
    def __init__(self, 
                 model,
                 trainset_size,
                 loss_fn: Callable, 
                 metrics_fn: Callable, 
                 logger,
                 config: Dict) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        
        self.logger = logger

        plm_params = [p for n,p in model.named_parameters() if 'encoder' in n]
        head_params = [p for n,p in model.named_parameters() if 'encoder' not in n]
        self.optim = AdamW([{'params': plm_params, 'lr':config.plm_lr}, 
                            {'params': head_params, 'lr':config.head_lr}], 
                            lr=config.plm_lr,
                            weight_decay=config.weight_decay
                          )
        
        training_step = int(config.num_epochs * (trainset_size / config.batch_size))
        warmup_step = int(config.warmup_ratio * training_step)  
        self.optim_schedule = get_linear_schedule_with_warmup(optimizer=self.optim, 
                                                              num_warmup_steps=warmup_step, 
                                                              num_training_steps=training_step)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
        
        self.print_every = int(config.print_every_ratio * trainset_size / config.batch_size)

        self.config = config

    def train(self, 
              model: nn.Module, 
              train_iter: DataLoader, 
              val_iter: DataLoader):
        model.train()
        if self.config.cuda and torch.cuda.is_available():
            model.cuda()
            pass
        
        best_res = [0, 0, 0]
        early_stop_cnt = 0
        best_state_dict = None
        step = 0
        for epoch in tqdm(range(self.config.num_epochs)):
            for batch in train_iter:
                inputs, offsets, heads, rels, masks = batch   
                
                if self.config.cuda and torch.cuda.is_available():
                    inputs_cuda = {}
                    for key, value in inputs.items():
                        inputs_cuda[key] = value.cuda()
                    inputs = inputs_cuda
    
                    offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
                
                arc_logits, rel_logits, loss = model(inputs, offsets, heads, rels, masks)
                
                self.optim.zero_grad()
                if self.config.cuda and self.config.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                else:
                    loss.backward()

                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config.grad_clip)

                if self.config.fp16:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()
                self.optim_schedule.step()

                metrics = self.metrics_fn(arc_logits, rel_logits, heads, rels, masks)

                if (step) % self.print_every == 0:
                    self.logger.info(f"--epoch {epoch}, step {step}, loss {loss}")
                    self.logger.info(f"  {metrics}")

                if val_iter is not None and self.config.eval_strategy == 'step' and (step + 1) % self.config.eval_every == 0:
                    avg_loss, uas, las = self.eval(model, val_iter)
                    res = [avg_loss, uas, las]
                    if las > best_res[2]:  # las
                        best_res = res
                        best_state_dict = model.state_dict()
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1
                    self.logger.info("--Best Evaluation: ")
                    self.logger.info("-loss: {}  UAS: {}  LAS: {} \n".format(*best_res))
                    # back to train mode
                    model.train()
                
                step += 1
                    
            if val_iter is not None and self.config.eval_strategy == 'epoch':
                avg_loss, uas, las = self.eval(model, val_iter)
                res = [avg_loss, uas, las]
                if las > best_res[2]:  # las
                    best_res = res
                    best_state_dict = model.state_dict()
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                self.logger.info("--Best Evaluation: ")
                self.logger.info("-loss: {}  UAS: {}  LAS: {} \n".format(*best_res))
                # back to train mode
                model.train()
                
            if early_stop_cnt >= self.config.num_early_stop:
                self.logger.info("--early stopping, training finished.")
                return best_res, best_state_dict

        self.logger.info("--training finished.")
        if best_state_dict is None:
            return 0.0, model.state_dict()
        return best_res, best_state_dict

    # eval func
    def eval(self, model: nn.Module, eval_iter: DataLoader, save_file: str = "", save_title: str = ""):
        model.eval()

        head_whole, rel_whole, mask_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
        arc_logit_whole, rel_logit_whole = torch.Tensor(), torch.Tensor()
        avg_loss = 0.0
        for step, batch in enumerate(eval_iter):
            inputs, offsets, heads, rels, masks = batch

            if self.config.cuda and torch.cuda.is_available():
                inputs_cuda = {}
                for key, value in inputs.items():
                    inputs_cuda[key] = value.cuda()
                inputs = inputs_cuda

                offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))
            
            with torch.no_grad():
                arc_logits, rel_logits = model(inputs, offsets, heads, rels, masks, evaluate=True)
                
            loss = self.loss_fn(arc_logits, rel_logits, heads, rels, masks)

            arc_logit_whole = torch.cat([arc_logit_whole, arc_logits.cpu()], dim=0)
            rel_logit_whole = torch.cat([rel_logit_whole, rel_logits.cpu()], dim=0)

            head_whole, rel_whole = torch.cat([head_whole, heads.cpu()], dim=0), torch.cat([rel_whole, rels.cpu()], dim=0)
            mask_whole = torch.cat([mask_whole, masks.cpu()], dim=0)

            avg_loss += loss.item() * len(heads)  # times the batch size of data

        metrics = self.metrics_fn(arc_logit_whole, rel_logit_whole, head_whole, rel_whole, mask_whole)
        uas, las = metrics['UAS'], metrics['LAS']

        avg_loss /= len(eval_iter.dataset)  # type: ignore

        self.logger.info("--Evaluation:")
        self.logger.info("Avg Loss: {}  UAS: {}  LAS: {} \n".format(avg_loss, uas, las))

        if save_file != "":
            results = [save_title, avg_loss, uas, las]  # type: ignore
            results = [str(x) for x in results]
            with open(save_file, "a+") as f:
                f.write(",".join(results) + "\n")  # type: ignore

        return avg_loss, uas, las  # type: ignore
    
    def save_results(self, save_file, save_title, results):
        saves = [save_title] + results
        saves = [str(x) for x in saves]
        with open(save_file, "a+") as f:
            f.write(",".join(saves) + "\n")  # type: ignore