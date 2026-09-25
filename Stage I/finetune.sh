#!/bin/bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1"


#Replace this with the appropriate path as needed.

config_file=/

pre_model=/
save_dir=/

mkdir -p $save_dir
data_dir=/
train_data=$data_dir/
test_data=$data_dir/
log_file=${save_dir}/train.log

accelerate launch --main_process_port 29502 --config_file ${config_file} train.py \
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
