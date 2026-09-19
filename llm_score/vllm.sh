export VLLM_LOG_LEVEL=ERROR
export VLLM_LOGGING_LEVEL=ERROR

model_name=score-8B1
step_num=1056
model_dir=/home/nian/ft_LLM/ft_model/TrainMethod_cqa/llm_score/save_models/${model_name}/step_${step_num}/Qwen3-8B-finetune
result_name=/home/nian/ft_LLM/ft_model/TrainMethod_cqa/llm_score/save_models/${model_name}/score_8_cqa_${step_num}.json

datasets_dir=/home/nian/ft_LLM/datasets/new_local/QAS/train
datasets_name=cqa_test1.jsonl   #test.json  #
CUDA_VISIBLE_DEVICES=6,7 python inference.py --model_dir $model_dir  --output_file $result_name  --data_file $datasets_dir/$datasets_name \
                         --rank 2 --gpu_memory_utilization 0.9  --bsz 256
