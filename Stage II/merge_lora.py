import argparse
import functools
import os

from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
#Replace this with the appropriate path as needed.
parser = argparse.ArgumentParser()
parser.add_argument("--lora_model", type=str, default="", help="Fine-tune the saved model path")
parser.add_argument('--output_dir', type=str, default='',    help="Save Directory for Merged Models")
parser.add_argument("--local_files_only", type=bool, default=False, help="Should the model be loaded locally only, without attempting to download it?")
args = parser.parse_args()
print(args)

assert os.path.exists(args.lora_model), f"Model Files{args.lora_model}Does not exist"

peft_config = PeftConfig.from_pretrained(args.lora_model)


device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map={"": "cpu"},
                                                             local_files_only=args.local_files_only)

model = PeftModel.from_pretrained(base_model, args.lora_model, local_files_only=args.local_files_only)

tokenizer = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path,
                                                            local_files_only=args.local_files_only)


model = model.merge_and_unload()
model.train(False)

if peft_config.base_model_name_or_path.endswith("/"):
    peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
save_directory = os.path.join(args.output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory, state_dict=model.state_dict(), max_shard_size='4GB')
tokenizer.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')
