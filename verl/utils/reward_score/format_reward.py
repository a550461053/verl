import re


def format_reward(solution_str):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    match = re.match(pattern, solution_str.strip(), re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0


def format_v1(solution_str):
    return format_reward(solution_str)
