import os
from datasets import load_dataset


class QARawDateset(object):

    def __init__(self, args):
        self.raw_datasets = load_dataset(
            'json',
            data_files={
                "train" : args.train_data,
                "eval" : args.eval_data,
            })

    def get_train_dataset(self):
        return self.raw_datasets["train"]

    def get_eval_dataset(self):
        return self.raw_datasets["eval"]


   
#We provide a prompt template for the Mohler dataset. For other datasets, simply replace the corresponding sections.
    def apply_prompt(self, example):
        question = example["question"]
        answer = example["answer"]
        score = example["human_score"]
        prompt = "<|im_start|>"  # "<｜begin▁of▁sentence｜>"

        prompt += "user\nTask description: You are a professional computer teacher. You will assign a reasonable score to answers regarding computer-related questions, with the score not exceeding 5 points."

        # prompt += standard

        prompt += f"question：{question}\n"

        prompt += f"Pending rating response：{answer}\nDirectly provide the final score. The output format is: \nFinal score: Score<|im_end|><|im_start|>assistant\nFinal score: {score}<|im_end|>"
        return prompt


if __name__ == '__main__':
    data_path = r"E:\code\ft_LLM\data"
    QARawDateset(seed=34, local_rank=-1, dataset_name=data_path)
