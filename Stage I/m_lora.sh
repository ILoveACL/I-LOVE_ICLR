export CUDA_VISIBLE_DEVICES="0,1"

#Replace this with the appropriate path as needed.
model_name=
step_num=
lora_model=/ /${model_name}/model.pt.step${step_num}
output_dir=/ /${model_name}/step_${step_num}
python merge_lora.py  \
    --lora_model  $lora_model \
    --output_dir  $output_dir
