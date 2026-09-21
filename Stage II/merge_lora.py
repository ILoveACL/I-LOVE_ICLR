import argparse
import functools
import os

from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lora_model", type=str, default="/home/nian/ft_LLM/ft_model/ft_llm/save_models/Qwen3-0.6B/model.pt.step1600", help="微调保存的模型路径")
parser.add_argument('--output_dir', type=str, default='/home/nian/ft_LLM/ft_model/ft_llm/save_models/Qwen3-0.6B/step_1600',    help="合并模型的保存目录")
parser.add_argument("--local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print(args)
# 检查模型文件是否存在
assert os.path.exists(args.lora_model), f"模型文件{args.lora_model}不存在"
# 获取Lora配置参数
peft_config = PeftConfig.from_pretrained(args.lora_model)

# 获取Whisper的基本模型
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map={"": "cpu"},
                                                             local_files_only=args.local_files_only)

# 与Lora模型合并
model = PeftModel.from_pretrained(base_model, args.lora_model, local_files_only=args.local_files_only)

tokenizer = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path,
                                                            local_files_only=args.local_files_only)

# 合并参数
model = model.merge_and_unload()
model.train(False)
# 保存的文件夹路径
if peft_config.base_model_name_or_path.endswith("/"):
    peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
save_directory = os.path.join(args.output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
os.makedirs(save_directory, exist_ok=True)
# 保存模型到指定目录中
model.save_pretrained(save_directory, state_dict=model.state_dict(), max_shard_size='4GB')
tokenizer.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')
