export CUDA_VISIBLE_DEVICES="6,7"

model_name=scorere_Qwen3-8B
step_num=600
lora_model=/home/nian/ft_LLM/ft_model/TrainMethod_re_mohler/llm_score_re/save_models/${model_name}/model.pt.step${step_num}
output_dir=/home/nian/ft_LLM/ft_model/TrainMethod_re_mohler/llm_score_re/save_models/${model_name}/step_${step_num}
python merge_lora.py  \
    --lora_model  $lora_model \
    --output_dir  $output_dir