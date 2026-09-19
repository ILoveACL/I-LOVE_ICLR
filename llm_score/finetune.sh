#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="6,7"

config_file=/home/nian/ft_LLM/ft_model/TrainMethod_cqa/llm_score/config.yaml

pre_model=/home/nian/pred_model/LLM_models/Qwen3-14B
save_dir=/home/nian/ft_LLM/ft_model/TrainMethod_cqa/llm_score/save_models/score-14B1

mkdir -p $save_dir
data_dir=/home/nian/ft_LLM/datasets/new_local/QAS/train
train_data=$data_dir/cqa_train1.jsonl
test_data=$data_dir/cqa_eval1.jsonl
log_file=${save_dir}/train.log

accelerate launch --main_process_port 29502 --config_file ${config_file} train.py \
        --train_data $train_data  \
        --eval_data $test_data   \
        --output_dir $save_dir \
        --pre_trained_model $pre_model \
        --train_batch_size 72  \
        --per_device_batch_size 4  \
        --max_length 512  \
        --num_train_epochs 4  \
        --seed 3407  \
        --log_interval 20  \
        --save_interval 20  \
        --gradient_checkpointing  \
        --lora  \
        --lr 3e-04   &> ${log_file}
