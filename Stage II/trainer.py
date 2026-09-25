import os
import re
import math
import torch
import logging
from Datasets import data_utils
from torch.optim import AdamW
from train_utils import to_device
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import (
    default_data_collator,
    get_scheduler,
)
from loss import ForCausalLMLoss

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

class Trainer:
    def __init__(
            self,
            args,
            rank,
            model,
            optimizer=None,
            scheduler=None,
            train_data=None,
            eval_data=None,
            processor=None,
            accelerator=None
    ):
        self.args = args
        self.rank = rank
        self.model=model
        self.config = model.config
        self.processor = processor
        self.accelerator = accelerator
        self.batch_size = args.train_batch_size

        self.steps = 0
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.device = accelerator.device
        self.loss_function = ForCausalLMLoss
        if train_data is None:
            self.train_data, self.eval_data = self.build_dataset()

        if args.train_batch_size > args.per_device_batch_size:
            self.gradient_accumulation_steps = args.train_batch_size // args.per_device_batch_size
        
        if optimizer is None:
            optimizer = AdamW(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.95), fused=True)
        if scheduler is None:
            scheduler = self.build_scheduler(optimizer)
        self.warp_model_optim_scheduler(model, optimizer, scheduler)

    def build_dataset(self):
        train_dataset, eval_dataset = data_utils.create_dataset(
            args=self.args, tokenizer=self.processor)

        train_sampler = RandomSampler(train_dataset)
        eval_sampler = RandomSampler(eval_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      collate_fn=default_data_collator,
                                      sampler=train_sampler,
                                      batch_size=self.args.per_device_batch_size)
        
        eval_dataloader = DataLoader(eval_dataset,
                                     collate_fn=default_data_collator,
                                     sampler=eval_sampler,
                                     batch_size=self.args.per_device_batch_size)
        return train_dataloader, eval_dataloader

    def build_scheduler(self, optimizer):
        num_update_steps_per_epoch = math.ceil(
            len(self.train_data) / self.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.num_train_epochs * num_update_steps_per_epoch,
        )
        return lr_scheduler

    def save_checkpoint(
        self,
        epoch,
        step=None
    ):
        os.makedirs(self.args.output_dir, exist_ok=True)
        if step is not None:
            ckpt_name = f"model.pt.step{self.steps}"
        else:
            ckpt_name = f"model.pt.ep{epoch}"
        filename = os.path.join(self.args.output_dir, ckpt_name)
        if self.accelerator.is_main_process:
            logging.info(f"Save checkpoint: {epoch}, step: {step}, total_step: {self.steps} to {filename}\n")
        self.model.eval()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            save_directory=filename, 
            is_main_process=True, 
            state_dict=self.accelerator.get_state_dict(self.model)
        )
        torch.cuda.empty_cache()

    def train(self):
        losses = self.validate_epoch(epoch=0)
        for epoch in range(self.args.num_train_epochs):
            if self.accelerator.is_main_process:
                logging.info(f"start training epoch: {epoch+1} / {self.args.num_train_epochs}")
            self.train_epoch(epoch)
            # losses = self.validate_epoch(epoch=epoch)
            # self.save_checkpoint(epoch, step=self.steps,)

    def train_epoch(self, epoch):
        for batch_idx, batch in enumerate(self.train_data):
            self.model.train()
            self.steps += 1
            batch = to_device(batch, self.device)
            loss = self.forward_step(batch)
            self.backward_step(loss, batch_idx+1)
            if self.steps % (self.log_interval*self.gradient_accumulation_steps) == 0:
                self.log(self.steps, loss, epoch, tag="train")
            if self.steps % (self.save_interval*self.gradient_accumulation_steps) == 0:
                losses = self.validate_epoch(epoch=epoch)
                self.save_checkpoint(epoch, step=self.steps,)
    
    def filter_score(self, input_strs, assis_labels=None):
        scores = []
        for i, l in enumerate(self.processor.batch_decode(input_strs)):
            score = re.findall(r"-?\d+\.?\d*", l)
            if len(score) > 0:
                scores.append(float(score[0]))
            else:
                scores.append((5-float(assis_labels[i])))  #5 is the maximum score for the Moher dataset; adjust this number as needed based on the score ranges of different datasets.
        return scores

    def loss_mse(self, pred_score, labels):
        mse_loss = torch.pow(pred_score-labels , 2)
        mse_loss = torch.sqrt(mse_loss)
        return mse_loss

    def get_re_loss(self, input_ids, en_loss, logits, token_index):
        score_labels = input_ids.clone()
        new_labels = score_labels[:, token_index:]
        scores_list = self.filter_score(new_labels)
        scores_l = torch.tensor(scores_list).to(logits)

        score_logits = logits[:, token_index-4:]
        en_loss_re = en_loss[:, token_index:-2]
        pred_ls = torch.argmax(score_logits, dim=-1)
        scores = self.filter_score(pred_ls, scores_list)
        scores = torch.tensor(scores).to(logits)
        re_adv = self.loss_mse(scores, scores_l).reshape(self.args.per_device_batch_size, 1)
        re_loss = torch.exp(en_loss_re - en_loss_re.detach()) * re_adv
        return re_loss

    def calcu_loss(self, batch, retval, en_loss):
        end_index = 77091
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        logits = retval["logits"]
        batch_indices, end_mask_indices = torch.where(input_ids == end_index)
        end_mask_indices += 4
        min_idx = end_mask_indices.min()
        re_loss = self.get_re_loss(input_ids, en_loss, logits, min_idx)
        mask = (labels != 151643)
        total_weight = re_loss.mean()+(en_loss.sum()/mask.sum())
        b_ = 0.1
        re_weight = (re_loss.mean()/total_weight)
        re_loss = (1-re_weight) * re_loss
        # re_loss = b_ * re_loss
        en_loss = re_weight * en_loss
        token_size = re_loss.size(1)
        # print("re_loss", re_loss.mean(-1))
        # print("en_loss_b", en_loss[:, min_idx:-2])
        en_loss[:, min_idx:-2] += re_loss
        # print("en_loss_a", en_loss[:, min_idx:-2])
        # exit(0)
        mask = (labels != 151643)
        loss = (en_loss.sum() / mask.sum()) #+ re_loss.mean()
        return loss

    def forward_step(self, batch):
        retval = self.model(**batch, ignore_index=151643, reduction='none')
        en_loss = self.loss_function(
            retval["logits"], 
            batch["labels"],
            vocab_size=self.config.vocab_size,
            ignore_index = 151643,
            reduction='none'
        )
        loss = self.calcu_loss(batch, retval, en_loss.reshape(self.args.per_device_batch_size, -1))
        return loss

    def backward_step(self, loss, batch_idx):
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        if batch_idx % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def validate_epoch(
        self,
        epoch
    ):
        if self.accelerator.is_main_process:
            logging.info(f"start evauating epoch: {epoch+1} / {self.args.num_train_epochs}")
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_data):
                batch = to_device(batch, self.device)
                loss = self.forward_step(batch)
                losses += loss.cpu().float()
            losses = losses / (batch_idx + 1)
            try:
                losses = get_all_reduce_mean(losses)
            except:
                pass
            if self.accelerator.is_main_process:
                self.log(self.steps, losses, epoch, tag="eval")
        return losses

    def log(
        self,
        step,
        loss,
        epoch,
        tag="train",
    ):
        loss = loss.float()
        description = (
            f"{tag}, "
            f"step: {step}, "
            f"epoch: {epoch+1}/{self.args.num_train_epochs}, "
            f"loss: {loss}, "
            f"rank: {self.rank}"
        )
        logging.info(description)

    def warp_model_optim_scheduler(self, model, optime, scheduler):
        self.model, self.train_data, self.eval_data, self.optimizer, self.scheduler = self.accelerator.prepare(
            model,
            self.train_data,
            self.eval_data,
            optime,
            scheduler
        )
