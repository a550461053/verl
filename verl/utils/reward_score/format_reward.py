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

import re


def format_reward(solution_str):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags,
    while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    match = re.match(pattern, solution_str.strip(), re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0


def format_v1(solution_str):
    return format_reward(solution_str)
