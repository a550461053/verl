# Copyright 2025 hilab team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.reward_score.format_reward import format_reward as format_compute
from verl.utils.reward_score.math_verify import compute_score as math_verify_compute


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 2) -> bool:
    math_verify_score = math_verify_compute(model_output, ground_truth, timeout_score)
    formatted_score = format_compute(model_output)
    if formatted_score == 1.0:
        total_score = (math_verify_score + formatted_score) / 2
    else:
        total_score = 0.0
    return {"format_sub": formatted_score, "math_verify_sub": math_verify_score, "score": total_score}
