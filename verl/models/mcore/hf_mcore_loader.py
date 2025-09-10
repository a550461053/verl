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

import json
import os
import re
from collections import defaultdict

import torch
from safetensors import safe_open
from tqdm import tqdm


class HfMcoreManager:
    def __init__(self):
        self.QKV_FUSE_NAME = ".linear_qkv."
        self.GATE_UP_FUSE_NAME = ".linear_fc1."
        self.UP_NAME = ".up_proj."
        self.Q_NAME = ".q_proj."
        self.REPLACE_DICT = {
            "up_proj.": "linear_fc1.",
            "down_proj.": "linear_fc2.",
            "model.embed_tokens.": "embedding.word_embeddings.",
            "model.layers.": "decoder.layers.",
            "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
            "self_attn.q_proj.": "self_attention.linear_q.",
            "self_attn.k_proj.": "self_attention.linear_k.",
            "self_attn.v_proj.": "self_attention.linear_v.",
            "self_attn.o_proj.": "self_attention.linear_proj.",
            "self_attn.q_layernorm.": "self_attention.q_layernorm.",
            "self_attn.k_layernorm.": "self_attention.k_layernorm.",
            "self_attn.kv_layernorm.": "self_attention.kv_layernorm.",
            # "self_attn.q_proj.": "self_attention.linear_q_proj.",
            "self_attn.q_a_proj.": "self_attention.linear_q_down_proj.",
            "self_attn.q_b_proj.": "self_attention.linear_q_up_proj.",
            "self_attn.kv_a_proj_with_mqa.": "self_attention.linear_kv_down_proj.",
            "self_attn.kv_b_proj.": "self_attention.linear_kv_up_proj.",
            "post_attention_layernorm.": "pre_mlp_layernorm.",
            ".gate_proj.": ".linear_gate.",
            ".gate.": ".router.",
            "model.norm.": "decoder.final_layernorm.",
            "lm_head.": "output_layer.",
        }

        self.split_mode_dict = {
            "self_attn.o_proj": (True, 1),  # RowParallel
            "down_proj": (True, 1),  # RowParallel
            "linear_q_up_proj": (True, 0),
            "self_attn.q_proj": (True, 0),
            "lm_head": (True, 0),
            "embed_tokens": (True, 0),
            "gate_proj": (True, 0),
            "up_proj": (True, 0),
        }

    def is_qkv_fusion(self, layer_name):
        suffix = ["weight", "bias"]
        for s in suffix:
            if self.Q_NAME + s in layer_name:
                return True
        return False

    def hf_to_mcore_name(self, layer_name):
        for key, value in self.REPLACE_DICT.items():
            layer_name = layer_name.replace(key, value)
        return layer_name

    def get_split_mode(self, layer_name):
        """
        Determines the split mode for a given layer name.

        Args:
            layer_name (str): The name of the layer to check.

        Returns:
            tuple: (is_split (bool), split_dim (int))
                - is_split: Whether the layer should be split.
                - split_dim: The dimension along which to split.
        """
        for mode in self.split_mode_dict.keys():
            suffix = [".weight", ".bias"]
            for s in suffix:
                if mode + s in layer_name:
                    return self.split_mode_dict[mode]

        return (False, 0)


class DeepseekV2HfLoader(HfMcoreManager):
    def __init__(
        self,
        config,
        model_config=None,
        tp_size=None,
        tp_rank=None,
        pp_size=None,
        pp_rank=None,
        ep_size=None,
        ep_rank=None,
    ):
        super().__init__()
        self.model_config = model_config
        self.model_path = config.model.path

        index_map_path = os.path.join(self.model_path, "model.safetensors.index.json")

        if os.path.exists(index_map_path):
            with open(index_map_path) as f:
                file_mapping = json.load(f)
                weight_mapping = file_mapping["weight_map"]
        else:
            model_tensor_path = os.path.join(self.model_path, "model.safetensors")
            with safe_open(model_tensor_path, framework="pt") as f:
                weight_mapping = {key: "model.safetensors" for key in f.keys()}

        layer_to_params = defaultdict(set)
        # layer_to_params ,key mglayername,value hflayernames
        for k, v in weight_mapping.items():
            if "model.embed_tokens" in k:
                layer_to_params["embedding"].add(k)
            elif "model.norm" in k:
                # final rms norm
                layer_to_params["final_norm"].add(k)
            elif "model.layers." in k:
                layer = int(k.split(".")[2])
                layer_to_params[layer].add(k)
            elif "lm_head.weight" in k:
                layer = "lm_head"
                layer_to_params[layer].add(k)
            else:
                raise AssertionError(k)

        self.hf_architecture = self.model_config.architectures[0]
        self.untie_embeddings_and_output_weights = not self.model_config.tie_word_embeddings
        self._weight_mapping = weight_mapping
        self._layer_to_params = layer_to_params
        self._tp_size = tp_size
        self._tp_rank = tp_rank
        self._pp_size = pp_size
        self._pp_rank = pp_rank
        self._ep_size = ep_size
        self._ep_rank = ep_rank
        self._num_layers = self.model_config.num_hidden_layers
        self._head_num = self.model_config.num_attention_heads
        self._num_query_groups = self.model_config.num_key_value_heads
        self._hidden_size = self.model_config.hidden_size
        self._ffn_hidden_size = self.model_config.intermediate_size
        self._qk_layernorm = self.model_config.qk_layernorm

        self._padded_vocab_size = self.model_config.vocab_size
        self._num_experts = self.model_config.n_routed_experts
        self._moe_ffn_hidden_size = self.model_config.moe_intermediate_size
        self._moe_shared_expert_intermediate_size = (
            self.model_config.n_shared_experts * self.model_config.moe_intermediate_size
        )

        self._kv_channel = int(self._hidden_size / self._head_num)

        self._first_pipeline_num_layers = getattr(config.actor.megatron, "first_pipeline_num_layers", None)
        self._last_pipeline_num_layers = getattr(config.actor.megatron, "last_pipeline_num_layers", None)

        self._multi_latent_attention = getattr(self.model_config, "multi_latent_attention", False)

        self._fuse_up_gate = True
        self._moe_grouped_gemm = True

    def adjust_mapping_table(self):
        if self._multi_latent_attention:
            # 在不使用mla时，目前的input_layernorm.weight 被映射为linear_qkv.layer_norm_weight。
            # 使用mla时该名字无需转换。
            if "input_layernorm.weight" in self.REPLACE_DICT:
                self.REPLACE_DICT.pop("input_layernorm.weight")
            # 添加mla norm的映射
            self.REPLACE_DICT["self_attn.kv_a_layernorm"] = "self_attention.kv_layernorm"
            self.REPLACE_DICT["self_attn.q_a_layernorm"] = "self_attention.q_layernorm"
        else:
            # 未启动mla时，需要做qkv fuse，不pop的话会在split分支中处理q（process_tensor_operations 中的 if split）
            self.split_mode_dict.pop("self_attn.q_proj")
            self.REPLACE_DICT["self_attn.linear_qkv"] = "self_attention.linear_qkv"

        # 当模型为qwen系列时，修改映射表。
        if "Qwen" in self.hf_architecture:
            # 修改attention之后的norm的映射，替换为了qwen的名字
            self.REPLACE_DICT["post_attention_layernorm.weight"] = "mlp.linear_fc1.layer_norm_weight"
            # 去掉旧的映射
            self.REPLACE_DICT.pop("post_attention_layernorm.")

        if not self._moe_grouped_gemm:
            self.REPLACE_DICT[".experts."] = ".experts.local_experts."

    def get_global_idx(self, layer_name, pp_offset, ep_offset):
        layer_name = self.get_global_layer_idx(layer_name, pp_offset)
        return self.get_global_expert_idx(layer_name, ep_offset)

    def load_safe_tensor_file(self, file_name, params):
        full_name = os.path.join(self.model_path, file_name)
        tensors = {}
        with safe_open(full_name, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in params:
                    tensor = f.get_tensor(key)
                    tensors[key] = tensor
        return tensors

    def _slice_mp(self, t, dim):
        if self._tp_size is None or self._tp_size == 1:
            return t
        full_size = list(t.shape)[dim]
        assert full_size % self._tp_size == 0
        split_size = full_size // self._tp_size
        return torch.split(t, split_size, dim=dim)[self._tp_rank].contiguous()

    def _load_layer_tensor(self, layer_name):
        assert layer_name in self._layer_to_params, f"{layer_name} not in {self._layer_to_params.keys()}"
        hf_layer_names = self._layer_to_params[layer_name]
        file_to_layers = defaultdict(set)
        tensors = {}
        for layer in hf_layer_names:
            file = self._weight_mapping[layer]
            file_to_layers[file].add(layer)
        for file_name, params in file_to_layers.items():
            tensors.update(self.load_safe_tensor_file(file_name, params))
        assert len(hf_layer_names) == len(tensors)
        return tensors

    def _handle_fuse_up_gate(self, gate_proj, up_proj, ffn_hidden_size, hidden_size):
        assert self._fuse_up_gate
        gate_up_proj = torch.cat([gate_proj, up_proj], dim=0)
        gate_up_proj = gate_up_proj.view(-1, ffn_hidden_size, hidden_size)
        gate_up_proj = self._slice_mp(gate_up_proj, dim=1).reshape(-1, hidden_size)
        return gate_up_proj.contiguous()

    def _handle_qkv(self, q, k, v):
        query_per_group = self._head_num // self._num_query_groups
        q_groups = torch.split(q, query_per_group * self._kv_channel, dim=0)
        k_groups = torch.split(k, self._kv_channel, dim=0)
        v_groups = torch.split(v, self._kv_channel, dim=0)
        fused_groups = [
            torch.cat([qg, kg, vg], dim=0) for qg, kg, vg in zip(q_groups, k_groups, v_groups, strict=False)
        ]
        fused = torch.cat(fused_groups, dim=0)
        fused = self._slice_mp(fused, 0)
        return fused

    def pop_non_local_experts(self, tensors):
        tensors_to_pop = []
        for key in tensors.keys():
            if ".experts." in key:
                num = int(key.split(".")[-3])
                if num < self.expert_begin or num >= self.expert_end:
                    tensors_to_pop.append(key)
        for non_local_expert in tensors_to_pop:
            tensors.pop(non_local_expert)

    def load_embedding_or_lm_head(self, name):
        tensors = self._load_layer_tensor(name)
        for key, tensor in tensors.items():  ## only 1 tensor
            embedding = tensor
            hf_name = key
        vocab_size, _ = embedding.shape
        if vocab_size >= self._padded_vocab_size:
            embedding = embedding[: self._padded_vocab_size, :]
        else:
            raise AssertionError(f"ckpt.vocab_size={vocab_size}, padded_vocab_size={self._padded_vocab_size}")
        tensors[hf_name] = embedding
        return tensors

    def calculate_begin_and_end(self, stage):
        # pp begin and end
        pp_size = self._pp_size if self._pp_size is not None else 1
        assert stage < pp_size
        if self._first_pipeline_num_layers is None and self._last_pipeline_num_layers is None:
            layer_per_stage = self._num_layers // pp_size
            begin = layer_per_stage * stage
            end = layer_per_stage * (stage + 1)
        else:
            first_last_layers = (
                self._first_pipeline_num_layers,
                self._last_pipeline_num_layers,
            )
            middle_num_stages = pp_size - sum([1 if x is not None else 0 for x in first_last_layers])

            middle_num_layers = self._num_layers - sum([x if x is not None else 0 for x in first_last_layers])
            middle_per_stage = middle_num_layers // middle_num_stages

            if self._first_pipeline_num_layers is None:
                self._first_pipeline_num_layers = middle_per_stage
            if self._last_pipeline_num_layers is None:
                self._last_pipeline_num_layers = middle_per_stage

            if stage == 0:
                begin = 0
                end = self._first_pipeline_num_layers
            elif stage == pp_size - 1:
                begin = self._num_layers - self._last_pipeline_num_layers
                end = self._num_layers
            else:
                begin = self._first_pipeline_num_layers + (stage - 1) * middle_per_stage
                end = begin + middle_per_stage
        assert begin < end
        self.begin = begin
        self.end = end

        # ep begin and end
        if self._num_experts is not None:
            assert self._num_experts % self._ep_size == 0
            num_local_expert = self._num_experts // self._ep_size
            self.expert_begin = self._ep_rank * num_local_expert
            self.expert_end = (self._ep_rank + 1) * num_local_expert
        else:
            self.expert_begin = None
            self.expert_end = None

    def process_tensor_operations(self, state_dict):
        tensors_adjusted = {}
        # special_layers = ("norm", "gate.", "_a_proj", "_b_proj")
        # split and fuse
        tensor_names = list(state_dict.keys())
        for hf_layer_name in tensor_names:
            if ".k_proj." in hf_layer_name or ".v_proj." in hf_layer_name or ".gate_proj." in hf_layer_name:
                # skip k_proj, v_proj, gate_proj
                continue

            tensor = state_dict[hf_layer_name]
            is_split, split_dim = self.get_split_mode(hf_layer_name)

            # non fuse but split tensor
            if is_split:
                tensor = self._slice_mp(tensor, split_dim)
                tensors_adjusted[hf_layer_name] = tensor.contiguous()

            # # non fuse and non split tensor
            # elif any(key in hf_layer_name for key in special_layers):
            #     tensors_adjusted[hf_layer_name] = tensor.contiguous()

            # qkv fuse
            if not self._multi_latent_attention and self.is_qkv_fusion(hf_layer_name):  # only query true
                k_name = hf_layer_name.replace(self.Q_NAME, ".k_proj.")
                v_name = hf_layer_name.replace(self.Q_NAME, ".v_proj.")
                qkv_fused = self._handle_qkv(
                    state_dict.pop(hf_layer_name), state_dict.pop(k_name), state_dict.pop(v_name)
                )
                mg_fuse_name = hf_layer_name.replace(self.Q_NAME, self.QKV_FUSE_NAME)
                tensors_adjusted[mg_fuse_name] = qkv_fused.contiguous()

            # gate up fuse
            elif self.UP_NAME in hf_layer_name:  # only up true
                gate_name = hf_layer_name.replace(self.UP_NAME, ".gate_proj.")
                _ffn_hidden_size = (
                    self._moe_ffn_hidden_size if "share" not in gate_name else self._moe_shared_expert_intermediate_size
                )
                _ffn_hidden_size = self._ffn_hidden_size if "expert" not in gate_name else _ffn_hidden_size
                gate_up_proj = self._handle_fuse_up_gate(
                    state_dict.pop(gate_name),
                    state_dict.pop(hf_layer_name),
                    _ffn_hidden_size,
                    self._hidden_size,
                )
                up_gate_out_name = hf_layer_name.replace(self.UP_NAME, self.GATE_UP_FUSE_NAME)
                tensors_adjusted[up_gate_out_name] = gate_up_proj.contiguous()
            else:
                tensors_adjusted[hf_layer_name] = tensor.contiguous()

        return tensors_adjusted

    def convert_layer_name(self, hf_name):
        # 专家索引调整
        if ".experts." in hf_name:
            hf_name = re.sub(
                r"\.experts\.(\d+)",
                lambda m: f".experts.{int(m.group(1)) - self.expert_begin}",
                hf_name,
            )
        # 基础名称转换
        mg_name = self.hf_to_mcore_name(hf_name)
        # 层号调整
        mg_name = re.sub(
            r"(decoder\.layers\.)(\d+)(\..*)",
            lambda m: f"{m.group(1)}{int(m.group(2)) - self.begin}{m.group(3)}",
            mg_name,
        )
        # Grouped GEMM特殊处理
        if self._moe_grouped_gemm and ".experts." in mg_name:
            mg_name = re.sub(r"\.experts\.(\d+)\.(.*?)\.weight$", r".experts.\2.weight\1", mg_name)
        return mg_name

    def refactor_tensors(self, state_dict):
        self.pop_non_local_experts(state_dict)
        # splite and fuse
        state_dict = self.process_tensor_operations(state_dict)

        # transfer hf name to mcore name
        layer_names = list(state_dict.keys())
        for layer_name in layer_names:
            mg_layer_name = self.convert_layer_name(layer_name)
            tensor = state_dict.pop(layer_name)
            state_dict[mg_layer_name] = tensor

        return state_dict

    def load_all_tensors(self):
        state_dict = {}
        for i in tqdm(range(self.begin, self.end)):
            tmp = self._load_layer_tensor(i)
            state_dict.update(tmp)

        if self.begin == 0:
            tmp = self.load_embedding_or_lm_head("embedding")
            assert len(tmp) == 1
            state_dict.update(tmp)

        if self.end == self._num_layers:
            tmp = self._load_layer_tensor("final_norm")
            state_dict.update(tmp)
            if self.untie_embeddings_and_output_weights:
                tmp = self.load_embedding_or_lm_head("lm_head")
                state_dict.update(tmp)
            elif self._pp_size > 1:
                tmp = self.load_embedding_or_lm_head("embedding")
                assert len(tmp) == 1
                state_dict.update(tmp)
                tensor = state_dict["model.embed_tokens.weight"]
                state_dict["lm_head.weight"] = tensor
                state_dict.pop("model.embed_tokens.weight")

        return state_dict

    def load(self):
        self.adjust_mapping_table()
        self.calculate_begin_and_end(self._pp_rank)
        state_dict = self.load_all_tensors()
        return self.refactor_tensors(state_dict)
