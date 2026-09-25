import json
from vllm import LLM
import torch
import os
import argparse

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["VLLM_LOG_LEVEL"] = "ERROR"
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"




def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default=r"") #Replace this with the appropriate path as needed.
    parser.add_argument("--output_file", type=str, default=r"") #Replace this with the appropriate path as needed.
    parser.add_argument("--data_file", type=str, default=r"")#Replace this with the appropriate path as needed.

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--trust_remote_code", type=bool, default=True)
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16", ], default="bf16")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--qas_type", type=str, default="mul")
    args = parser.parse_args()
    return args

def load_model(args):
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    llm = LLM(model=args.model_dir,  # LLM模型位置
              tensor_parallel_size=args.rank,  # 使用几张GPU卡进行推理
              trust_remote_code=args.trust_remote_code,
              dtype=dtype,  # 模型参数类型
              max_model_len=args.max_length,  # prompt和生产的最大token数
              seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization  # GPU显存使用率
              )
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = args.temperature  # 温度，控制模型生成
    sampling_params.top_p = args.top_p  # 生成结果的采样概率
    sampling_params.top_k = args.top_k  # 生成结果的采样大小
    sampling_params.max_tokens = args.max_length  # prompt和生产的最大token数
    return llm, sampling_params

def read_file(file_path):
    datasets = []
    with open(file_path, 'r', encoding='utf-8') as r:
        for line in r.readlines():
            data = json.loads(line)
            datasets.append(data)
    return datasets


#We provide a prompt template for the Mohler dataset. For other datasets, simply replace the corresponding sections.
def apply_prompt(example):
    question = example["question"]
    answer = example["answer"]
    score = example["human_score"]
    prompt = "<|im_start|>"  # "<｜begin▁of▁sentence｜>"

    prompt += "user\nTask description: You are a professional computer teacher. You will assign a reasonable score to answers regarding computer-related questions, with the score not exceeding 5 points."

    # prompt += standard

    prompt += f"question：{question}\n"

    prompt += f"Pending rating response：{answer}\nDirectly provide the final score. The output format is: \nFinal score: Score<|im_end|><|im_start|>assistant\nFinal score: "
    return prompt

def main(args):

    mdoel, sp = load_model(args)
    ds = read_file(args.data_file)
    prompts = []
    for qa in ds:
        prompt = apply_prompt(qa)
        prompts.append(prompt)
    num_prompt = len(prompts)
    bsz = args.bsz
    n = num_prompt // bsz
    with open(args.output_file, "w", encoding="utf-8") as w:
        for i in range(n+1):
            if (i + 1) * bsz < num_prompt:
                sub_prompt = prompts[i * bsz:(i + 1) * bsz]
            else:
                sub_prompt = prompts[i * bsz:]
            outputs = mdoel.generate(sub_prompt, sp)
            for j, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                pt = output.prompt
                write_text = {
                    "question": ds[i * bsz + j]["question"],
                    "answer": ds[i * bsz + j]["answer"],
                    "human": ds[i * bsz + j]["human_score"],
                    "llm": generated_text
                }
                w.write(json.dumps(write_text, ensure_ascii=False))
                w.write("\n")

if __name__ == '__main__':
    args = argument()
    main(args)
