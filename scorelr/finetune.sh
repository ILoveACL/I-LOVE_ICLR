#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="6,7"

config_file=/home/nian/ft_LLM/ft_model/TrainMethod_re_mohler/llm_score_re/config.yaml

pre_model=/home/nian/pred_model/LLM_models/Qwen3-8B
save_dir=/home/nian/ft_LLM/ft_model/TrainMethod_re_mohler/llm_score_re/save_models/scorere_Qwen3-8B

mkdir -p $save_dir
data_dir=/home/nian/ft_LLM/datasets/new_local/QAS/train
train_data=$data_dir/mohler_train.jsonl
test_data=$data_dir/mohler_eval.jsonl
log_file=${save_dir}/train.log

accelerate launch --main_process_port 29501 --config_file ${config_file} train.py \
        --train_data $train_data  \
        --eval_data $test_data   \
        --output_dir $save_dir \
        --pre_trained_model $pre_model \
        --train_batch_size 24  \
        --per_device_batch_size 4  \
        --max_length 512  \
        --num_train_epochs 8  \
        --seed 3407  \
        --log_interval 20  \
        --save_interval 20  \
        --gradient_checkpointing  \
        --lora  \
        --lr 3e-04   &> ${log_file}


