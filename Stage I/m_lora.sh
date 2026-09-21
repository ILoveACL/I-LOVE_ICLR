export CUDA_VISIBLE_DEVICES="6,7"


model_name=score-14B1
step_num=2112
lora_model=/home/nian/ft_LLM/ft_model/TrainMethod_cqa/llm_score/save_models/${model_name}/model.pt.step${step_num}
output_dir=/home/nian/ft_LLM/ft_model/TrainMethod_cqa/llm_score/save_models/${model_name}/step_${step_num}
python merge_lora.py  \
    --lora_model  $lora_model \
    --output_dir  $output_dir