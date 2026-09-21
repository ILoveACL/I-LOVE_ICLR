#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="4,5,6,7"

config_file="/home/nian/ft_LLM/ft_llm/config.yaml"

pre_model="/home/nian/pred_model/LLM_models/DeepSeek-R1-Distill-Qwen-32B"

save_dir='/home/nian/ft_LLM/ft_llm/save_models'
mkdir $save_dir

data_dir="/home/nian/ft_LLM/datasets/new_local/QAS/train"
for tst in train_sin_qa.json train.json train_inf_qa.json train_mul_qa.json; do
    train_file=$data_dir/$tst
    test_file=$data_dir/${tst/train/test}
    key=${tst:6:3}
    sub_save_dir=$save_dir/$key
    mkdir $sub_save_dir
    log_file="${sub_save_dir}/log.txt"s
    accelerate launch --config_file ${config_file} train.py \
        --output_dir $sub_save_dir \
        --pre_trained_model $pre_model \
        --train_data $train_file  \
        --eval_data $test_file  \
        --num_train_epochs 10   &> ${log_file}
done

