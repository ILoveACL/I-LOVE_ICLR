import os
import logging
from trainer import Trainer
import train_utils
from peft import  LoraConfig, TaskType
from argument_utils import argument
from accelerate import Accelerator
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

deepspeed_logger = logging.getLogger("DeepSpeed")
deepspeed_logger.setLevel(logging.WARNING)
for hdl in deepspeed_logger.handlers:
    hdl.setLevel(logging.WARNING)
    
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_lora_config(args):
    if args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=args.target_modules,
            inference_mode=args.inference_mode,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    else:
        lora_config = None
    return lora_config

def main():
    args = argument()
    lora_config = get_lora_config(args)
    train_utils.set_all_seed(args.seed)
    rank = int(os.environ.get("RANK", 0))
    accelerator = Accelerator()   
    args.device = accelerator.device
    processor = train_utils.LoadProcessor(args)
    model = train_utils.load_model(args, lora_config)
    if accelerator.is_main_process:
        logger.info(args)
        logger.info(model)
    #     for name, param in model.named_parameters():
    #         logger.info(f"{name} : {param.requires_grad}")
    trainer = Trainer(
            args,
            rank,
            model,
            processor=processor,
            accelerator=accelerator)
    trainer.train()
    accelerator.end_training()



if __name__ == '__main__':
    main()