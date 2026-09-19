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
        info = example["info"]
        question = example["question"]
        answer = example["answer"]
        score = example["human_score"]
        prompt = "<|im_start|>" #"<｜begin▁of▁sentence｜>"
        if info is not None:
            prompt += "user\n任务说明：你是一名中文语言学专家，给定下列评分标准，根据评分标准对根据内容回答问题的答案进行评分，给出最终得分，总分为8分。"
        else:
            prompt += "user\n任务说明：你是一名中文语言学专家，给定下列评分标准，根据评分标准对回答问题的答案进行评分，给出最终得分，总分为8分。"
        prompt += standard
        if info is not None:
            prompt += f"内容：{info}\n"
        prompt += f"问题：{question}\n"
        ex_str = self.get_example(example)
        if ex_str is not None:
            prompt += ex_str
        prompt += f"待评分回答：{answer}\n直接给出最终得分。输出格式为：\n最终得分：分数<|im_end|><|im_start|>assistant\n最终得分：{score}<|im_end|>"
        return prompt


if __name__ == '__main__':
    data_path = r"E:\code\ft_LLM\data"
    QARawDateset(seed=34, local_rank=-1, dataset_name=data_path)