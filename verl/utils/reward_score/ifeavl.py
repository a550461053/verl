import json

import fire
import numpy as np
import pandas as pd
from lm_eval.tasks.ifeval.utils import process_results
from tqdm import tqdm


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
    if "Assistant: <think>" in solution_str:
        user_prompt = solution_str.split("Assistant: <think>")[0].split("User: ")[1].strip()
        new_response = solution_str.split("Assistant: <think>")[1].split("<｜end▁of▁sentence｜>")[0].strip()
        # 将思考过程去除
        if "</think>" in new_response:
            new_response = new_response.split("</think>")[1].strip()
        return user_prompt, new_response
    else:
        user_prompt, response = solution_str.split("User: ")[1].split("Assistant: ").strip()
        user_prompt = user_prompt.split("<|im_end|>")[0].strip()
        response = response.split("<｜end▁of▁sentence｜>")[0].strip()
        # 将思考过程去除
        if "</think>" in response:
            response = response.split("</think>")[1].strip()
        return user_prompt, response


def compute_score(
    solution_str,
    ground_truth,
):
    try:
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        prompt, response = "", solution_str
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
    except Exception as err:
        print(f"IFEval Reward, [ERROR] Compute Score Error: {err}")
        return 0.0


def debug(test_file: str = "/cpfs/user/liuyanjiang/Eng/unitest/offline_single_res.jsonl"):
    df = pd.read_parquet(
        "/cpfs/user/lizichao/RedMOE-verl-new/dataset/test_ifeval_benchmark.parquet",
    )
    content = [{col: row[col] for col in df.columns} for _, row in df.iterrows()]

    prompt_to_gt = dict()
    for item in tqdm(content):
        prompt_to_gt[item["prompt"][1]["content"]] = convert_arrays_and_floats_to_simple_types(
            item["reward_model"]["ground_truth"]
        )

    with open(test_file) as f:
        content = [json.loads(line) for line in f]

    print("read done")

    """
    acc = 0
    processed_prompts = set()
    for item in tqdm(content):
        if item['prompt'][1]['content'] in processed_prompts:
            continue
        gt = prompt_to_gt[item['prompt'][1]['content']]
        x = {"key": None, "instruction_id_list": gt["instruction_id_list"],
                 "kwargs": gt["kwargs"], "prompt": item['prompt'][1]['content']}
        ifeval_results = process_results(x, item['rollout'])
        if ifeval_results['prompt_level_strict_acc']:
            acc += 1
        processed_prompts.add(item['prompt'][1]['content'])

    print(acc / len(processed_prompts))
    """
    acc = 0
    processed_prompts = set()
    for item in tqdm(content):
        prompt = item["prompt"].split("<|userprompt|>")[1].split("<|endofuserprompt|>")[0]
        if prompt in processed_prompts:
            continue
        gt = prompt_to_gt[prompt]
        x = {"key": None, "instruction_id_list": gt["instruction_id_list"], "kwargs": gt["kwargs"], "prompt": prompt}
        ifeval_results = process_results(x, [item["resp"]])
        if ifeval_results["prompt_level_strict_acc"]:
            acc += 1
        processed_prompts.add(prompt)

    print(acc / len(processed_prompts))
    print(len(processed_prompts))


if __name__ == "__main__":
    fire.Fire(debug)
