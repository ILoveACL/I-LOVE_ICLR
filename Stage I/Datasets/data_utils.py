import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, Subset
from deepspeed import get_accelerator
from Datasets.qa_dataset import QARawDateset

mask_token = "assistant"#"<｜Assistant｜>"
token_ids = 872
# ignore_index = -100
ignore_index = 151643
def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx

class PromptDataset(Dataset):

    def __init__(self, prompt_dataset) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset

    def __len__(self):
        length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        return {
            "input_ids": self.prompt_dataset[idx]["input_ids"],
            "attention_mask": self.prompt_dataset[idx]["attention_mask"],
            "labels": self.prompt_dataset[idx]["labels"],
        }

def process_dataset(dataset, raw_dataset, tokenizer, max_seq_len, mask_idx):
    prompt_dataset = []
    for i, tmp_data in enumerate(dataset):
        chosen_sentence = raw_dataset.apply_prompt(tmp_data)
        chosen_token = tokenizer(
            chosen_sentence,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            padding_side='left'
        )
        input_ids = chosen_token["input_ids"].squeeze(0)
        chosen_token["input_ids"] = input_ids
        chosen_token["attention_mask"] = chosen_token["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        end_mask_indices = torch.where(input_ids == mask_idx)[0]
        end_mask_indices = end_mask_indices + 5
        need_mask_indices = end_mask_indices
        labels[:need_mask_indices] = ignore_index 
        chosen_token["labels"] = labels
        prompt_dataset.append(chosen_token)

    return PromptDataset(prompt_dataset)

def create_prompt_dataset(
        args,
        tokenizer,
):
    raw_dataset = QARawDateset(args)
    mask_idx = tokenizer.encode(mask_token)[-1]
    train_dataset = process_dataset(
        raw_dataset.get_train_dataset(),
        raw_dataset,
        tokenizer,
        args.max_length,
        mask_idx
    )
    eval_dataset = process_dataset(
        raw_dataset.get_eval_dataset(),
        raw_dataset,
        tokenizer,
        args.max_length,
        mask_idx
    )
    return train_dataset, eval_dataset

def create_dataset(
        args,
        tokenizer,
        reload=False
):
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    fname = f"seed{args.seed}"
    train_fname = f"{output_path}/train_{fname}.pt"
    eval_fname = f"{output_path}/eval_{fname}.pt"
    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).to(
        get_accelerator().current_device_name())
    torch.distributed.all_reduce(buf_create_cache)

    if (buf_create_cache.item() != 0 or reload):
        train_dataset, eval_dataset = create_prompt_dataset(
            args,
            tokenizer,
        )
        shuffle_idx = get_shuffle_idx(args.seed, len(train_dataset))
        train_dataset = Subset(train_dataset, shuffle_idx.tolist())
        shuffle_idx = get_shuffle_idx(args.seed, len(eval_dataset))
        eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    torch.distributed.barrier()
    return (torch.load(train_fname,weights_only=False),
            torch.load(eval_fname,weights_only=False))
