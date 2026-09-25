export VLLM_LOG_LEVEL=ERROR
export VLLM_LOGGING_LEVEL=ERROR
#Replace this with the appropriate path as needed.
model_name=
step_num=
model_dir=/ /${model_name}/step_${step_num}/Qwen3-8B-finetune
result_name=/ /${model_name}/score_8_cqa_${step_num}.json

datasets_dir=/
datasets_name=test.jsonl   #test.json  #
CUDA_VISIBLE_DEVICES=0,1 python inference.py --model_dir $model_dir  --output_file $result_name  --data_file $datasets_dir/$datasets_name \
                         --rank 2 --gpu_memory_utilization 0.9  --bsz 256
