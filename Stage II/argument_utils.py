import argparse
from transformers import SchedulerType

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=r"/home/nian/ft_LLM/datasets/new_local/train/train.json")
    parser.add_argument("--eval_data", type=str, default=r"/home/nian/ft_LLM/datasets/new_local/train/test.json")
    parser.add_argument("--output_dir", type=str, default=r"/home/nian/ft_LLM/ft_LLM/save_models")
    parser.add_argument("--pre_trained_model", type=str, default=r"/home/nian/pred_model/LLM_models/DeepSeek-R1-Distill-Qwen-7B")

    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bit", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=1)

    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
                        choices = ["linear", "cosine", "cosine_with_restarts", "polynomial",
                                   "constant", "constant_with_warmup"],)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--lora", action='store_true')
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--target_modules", default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj",])  # "up_proj", "gate_proj", "down_proj",
    parser.add_argument("--inference_mode", type=bool, default=False)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    return args
