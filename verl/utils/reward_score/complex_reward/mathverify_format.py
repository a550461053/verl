from verl.utils.reward_score.math_verify import compute_score as math_verify_compute
from verl.utils.reward_score.format_reward import format_reward as format_compute 

def compute_score(model_output: str, ground_truth: str, timeout_score: float = 2) -> bool:
    math_verify_score = math_verify_compute(model_output, ground_truth, timeout_score)
    formatted_score = format_compute(model_output)
    if formatted_score == 1.0:
        total_score = (math_verify_score+formatted_score)/2
    else:
        total_score = 0.0
    return {'format_sub': formatted_score, 'math_verify_sub': math_verify_score, 'score': total_score}

