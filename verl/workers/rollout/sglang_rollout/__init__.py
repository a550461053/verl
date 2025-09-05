# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import multiprocessing as mp
from sglang.srt.entrypoints.engine import mp as sglang_mp
from .sglang_rollout import SGLangRollout


class PatchedProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        self.registe_dots_model()
        super().run()

    def registe_dots_model(self):
        from verl.models.dots.sglang import XdgMoEForCausalLM
        from sglang.srt.models.registry import ModelRegistry
        ModelRegistry.models["XdgMoEForCausalLM"] = XdgMoEForCausalLM

sglang_mp.Process = PatchedProcess

__all__ = ["SGLangRollout"]