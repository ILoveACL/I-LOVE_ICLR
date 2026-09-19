import torch
import random
import numpy as np
from peft import get_peft_model
from deepspeed import get_accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed
)

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def LoadProcessor(args):
    processor = AutoTokenizer.from_pretrained(
        args.pre_trained_model,
        trust_remote_code=True
    )
    return processor

def load_model(args, lora_config):
    bit_config = None
    if args.use_bit:
        bit_config = set_bit_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.pre_trained_model, 
        quantization_config=bit_config,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
    return model

def set_bit_config():
    bit_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return bit_config

def set_all_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    get_accelerator().manual_seed_all(seed)