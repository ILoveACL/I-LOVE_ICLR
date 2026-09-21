import os
from datasets import load_dataset

standard = """评分标准：
    三档（7—8分）：能就所有问题有效作答；能使用一个或多个完整的句子进行表达，所用语句在形式和内容上达到了较高水平；回答内容切题，表达清楚；回答中可以偶有停延、重复和语言错误。
    二档（4—6分）：能用结构简单的句子回答听到的所有问题，且内容基本切题，可以有多处停延、重复和语法错误，但基本能达意；或者只回答了部分问题，但内容切题，表达清楚，停延、重复和语言错误较少。
    一档（1—3分）：能勉强回答部分问题；不能说出完整的句子，表述以零散的词语为主，停延、重复较多，但说出的内容与听到的问题有关联。
    0档（0分）：不能作答；或答非所问；或作答的内容完全不可理解。\n"""

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

    def get_example(self, examples):
        example_list = examples.get("example", None)

        if example_list is None:
            return None
        else:
            example_str = "评分示例:\n"
            for i, k in enumerate(example_list):
                example_str += f"  示例{i + 1}:\n    回答：{k[0]}\n    最终得分：{k[1]}\n"
        return example_str

    def apply_prompt(self, example):
        question = example["question"]
        answer = example["answer"]
        score = example["human_score"]
        prompt = "<|im_start|>"  # "<｜begin▁of▁sentence｜>"

        prompt += "user\nTask description: You are a professional computer teacher. You will assign a reasonable score to answers regarding computer-related questions, with the score not exceeding 5 points."

        #prompt += standard

        prompt += f"question：{question}\n"
        ex_str = self.get_example(example)
        if ex_str is not None:
            prompt += ex_str
        prompt += f"Pending rating response：{answer}\nDirectly provide the final score. The output format is: \nFinal score: Score<|im_end|><|im_start|>assistant\nFinal score: {score}<|im_end|>"
        return prompt


if __name__ == '__main__':
    data_path = r"E:\code\ft_LLM\data"
    QARawDateset(seed=34, local_rank=-1, dataset_name=data_path)