export VLLM_LOG_LEVEL=ERROR
export VLLM_LOGGING_LEVEL=ERROR

model_name=scorere_Qwen3-8B_lr
step_num=360
model_dir=/home/nian/ft_LLM/ft_model/TrainMethod_re_mohler/llm_score_re/save_models/${model_name}/step_${step_num}/Qwen3-8B-finetune-finetune
result_name=/home/nian/ft_LLM/ft_model/TrainMethod_re_mohler/llm_score_re/save_models/${model_name}/scorerelr_8_mohler_${step_num}.json

datasets_dir=/home/nian/ft_LLM/datasets/new_local/QAS/train
datasets_name=mohler_test.jsonl #test.json  
CUDA_VISIBLE_DEVICES=6,7 python inference.py --model_dir $model_dir  --output_file $result_name  --data_file $datasets_dir/$datasets_name \
                         --rank 2 --gpu_memory_utilization 0.9  --bsz 256
