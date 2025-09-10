import json

import numpy as np
from lm_eval.tasks.ifeval.utils import process_results


def convert_arrays_and_floats_to_simple_types(obj):
    """
    递归地将对象中：
    1. 所有的 NumPy array 转换为普通 Python 列表
    2. 所有表示整数的浮点数转换为整数
    """
    if isinstance(obj, np.ndarray):
        # 如果对象是 NumPy array，转换它
        return [convert_arrays_and_floats_to_simple_types(item) for item in obj]
    elif isinstance(obj, (list, tuple)):
        # 如果是列表或元组，递归处理每个元素
        return [convert_arrays_and_floats_to_simple_types(item) for item in obj]
    elif isinstance(obj, dict):
        # 如果是字典，递归处理每个值
        return {key: convert_arrays_and_floats_to_simple_types(value) for key, value in obj.items()}
    elif isinstance(obj, float) and obj.is_integer():
        # 如果是浮点数且值为整数，转换为整数
        return int(obj)
    else:
        # 其他类型原样返回
        return obj


# Qwen-2.5-Instruct
def split_solution_qwen(solution_str):
    user_prompt, response = solution_str.split("<|im_start|>user")[1].split("<|im_start|>assistant")
    user_prompt = user_prompt.split("<|im_end|>")[0].strip()
    response = response.split("<|im_end|>")[0].strip()
    return user_prompt, response


# Qwen-2.5-R1-Distilled


def split_solution_qwen_distilled(solution_str):
    user_prompt = solution_str.split("<｜Assistant｜><think>")[0].split("<｜begin▁of▁sentence｜><｜User｜>")[1].strip()
    new_response = solution_str.split("<｜Assistant｜><think>")[1].split("<｜end▁of▁sentence｜>")[0].strip()
    return user_prompt, new_response


def split_r1_or_qwen(solution_str):
    if "<|userprompt|>" in solution_str and "<|endofuserprompt|>" in solution_str:
        user_prompt, response = solution_str.split("<|userprompt|>")[1].split("<|endofuserprompt|><|response|>")
        response = response.split("<|endofresponse|>")[0].strip()
    else:
        user_prompt = ""
        if "<|endofuserprompt|><|response|>" in solution_str:
            response = solution_str.split("<|endofuserprompt|><|response|>")[1]
            if "<|endofresponse|>" in response:
                response = response.split("<|endofresponse|>")[0].strip()
        elif "<|endofresponse|>" in solution_str:
            response = solution_str.split("<|endofresponse|>")[0].strip()
        else:
            response = solution_str
    # 将思考过程去除
    if "</think>" in response:
        response = response.split("</think>")[1].strip()
    return user_prompt, response


def compute_score(
    solution_str,
    ground_truth,
):
    # try:
    # print(f"=== {solution_str=} ===")
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    prompt, response = split_r1_or_qwen(solution_str)
    item = {
        "key": None,
        "instruction_id_list": convert_arrays_and_floats_to_simple_types(ground_truth["instruction_id_list"]),
        "kwargs": convert_arrays_and_floats_to_simple_types(ground_truth["kwargs"]),
        "prompt": prompt,
    }

    ifeval_results = process_results(item, [response])
    if ifeval_results["prompt_level_strict_acc"]:
        return 1.0
    else:
        return 0.0
    # except Exception as err:
    #     print(f'IFEval Reward, [ERROR] Compute Score Error: {err}')
    #     return 0.
