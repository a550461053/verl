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

import asyncio
import gc
import os
import threading
import time
from typing import Any

import torch

from verl.trainer.ppo.pipeline.pipeline_utils import enhanced_print
from verl.utils.megatron_utils import per_tensor_generator
from verl.workers.param_update.ray_async_communication import (
    cross_process_ray_get,
    cross_process_ray_put,
)


# ============================================================================
# ParamUpdateManager class
# ============================================================================
class ParamUpdateManager:
    def __init__(
        self,
        model_params,
        model_config,
        weight_converter,
        transformer_config,
        layer_name_mapping,
        convert_qkv_gate_up_by_simple_split,
        enable_async_rl: bool = True,  # New: whether to enable async RL optimization
        target_device: str = "cuda",  # Modified: default to GPU for sync mode
        enable_param_async: bool = False,  # Modified: True=CPU async, False=NCCL GPU sync
        store_refs_queue=None,  # New: store refs queue
        param_update_preduce_bucket_size_mb: int = 512,  # New: parameter preprocessing bucket size
        enable_fused_communication: bool = True,  # New: whether to use fused communication (flatten+concatenate)
    ):
        # Basic configuration
        self.model_params = model_params
        self.model_config = model_config
        self.weight_converter = weight_converter
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.target_device = target_device
        self.enable_param_async = enable_param_async
        self.store_refs_queue = store_refs_queue

        # Async RL optimization switch
        self.enable_async_rl = enable_async_rl

        # Communication optimization switch
        self.enable_fused_communication = enable_fused_communication

        # Log control switch
        self.verbose_logging = os.environ.get("PARAM_UPDATE_VERBOSE_LOG", "false").lower() == "true"

        # Debug switch: skip func_call execution
        self.skip_func_call = os.environ.get("SKIP_FUNC_CALL", "0") == "1"

        # Set two bucket sizes separately
        # 1. Send stage bucket fusion granularity (train stage memory) - can be set larger
        self.send_bucket_size_mb = param_update_preduce_bucket_size_mb

        enhanced_print(
            "ParamUpdateManager",
            None,
            f"Bucket sizes: send={self.send_bucket_size_mb}MB, "
            f"sync_mode={'CPU_async' if enable_param_async else 'NCCL_GPU_sync'}",
        )

        # Initialize parameter metadata and groups
        self._params_meta = []
        self._param_groups = {}

        # Initialize parameter update completion flag for NCCL sync mode
        self._param_update_completed = True  # Start as completed

        # Set default Ray collective name
        self.ray_col_name = (
            "actor_rollout_sync"  # Dedicated parameter sync group name, avoid conflict with SGLang communication
        )

        # Initialize async RL optimization components
        if self.enable_async_rl:
            self._init_async_rl_components()

        self._debug_mode = False  # Debug mode switch

        if self.skip_func_call:
            enhanced_print(
                "param_update", None, "⚠️ SKIP_FUNC_CALL enabled - func_call execution will be skipped for debugging"
            )

    def setup_for_queue(self, queue):
        self.store_refs_queue = queue

    def _init_async_rl_components(self):
        """Initialize async RL optimization components"""
        # Version counter
        self.current_version = -1

        enhanced_print("ParamUpdateManager", None, "Initialized async RL components")

    def get_async_rl_stats(self) -> dict[str, Any]:
        """Get async RL statistics"""
        if not self.enable_async_rl:
            return {"enabled": False}

        return {"enabled": True, "current_version": self.current_version}

    def set_model_params(self, model_params):
        """Set model parameters - for dynamic setting"""
        self.model_params = model_params
        enhanced_print("param_update", None, f"Model params set: {type(model_params)}")

    def is_train_node(self) -> bool:
        """Check if it's a training node"""
        return self.model_params is not None

    def is_train_master_node(self):
        """Check if it's a training master node"""
        return self.is_train_node() and self.rank == 0

    def is_generation_node(self) -> bool:
        """Check if it's a generation node"""
        return self.model_params is None

    def is_generation_master_node(self):
        """Check if it's a generation master node"""
        return self.is_generation_node() and self.is_engine_master

    def engine_idx(self):
        """for global queue"""
        return (self.rank - self.rank_offset) // self.engine_tp_size

    def one_to_all_nums(self):
        """for global queue"""
        return self.engine_nums

    def setup_for_ray_col(
        self,
        rank: int,
        size: int,
        rank_offset: int,
        engine_nums: int,
        engine_tp_size: int,
        is_engine_master: bool,
        name: str,
        backend: str = "nccl",
    ):
        """Setup Ray communication - choose different communication methods based on parameter sync mode"""
        try:
            self.ray_col_name = name
            self.rank = rank
            self.size = size
            self.rank_offset = rank_offset
            self.engine_nums = engine_nums
            self.is_engine_master = is_engine_master
            self.engine_tp_size = engine_tp_size

            # Choose communication method based on parameter async mode
            if self.enable_param_async:
                # CPU async mode: use Ray put/get
                enhanced_print(
                    "param_update",
                    None,
                    f"Ray put/get CPU async communication ready: rank={rank}, size={size}, name={name}",
                )
            else:
                # NCCL GPU sync mode: use Ray Collective
                from ray.util.collective import init_collective_group

                init_collective_group(
                    group_name=name,
                    world_size=size,
                    rank=rank,
                    backend=backend,
                )
                enhanced_print(
                    "param_update",
                    None,
                    f"Ray Collective NCCL GPU sync ready: rank={rank}, size={size}, name={name}, backend={backend}",
                )

                # For NCCL GPU sync mode, also setup train-generate sync group
                # This will be used for parameter synchronization between train and generate nodes
                self.train_generate_sync_group = name  # Use the same group for parameter sync

            return True

        except Exception as e:
            enhanced_print("param_update", None, f"Failed to setup Ray communication: {e}")
            return False

    def setup_train_generate_sync_group(self, train_ranks: list[int], generate_ranks: list[int]):
        """Setup NCCL communication group between train and generate nodes - all ranks participate"""
        try:
            # Create sync group between all train and generate nodes
            sync_group_name = "train_generate_sync"
            sync_ranks = train_ranks + generate_ranks
            sync_size = len(sync_ranks)

            # Find current rank in sync group
            sync_rank = -1
            if self.rank in train_ranks:
                # Train node: find position in train_ranks
                for i, rank in enumerate(train_ranks):
                    if rank == self.rank:
                        sync_rank = i
                        break
            elif self.rank in generate_ranks:
                # Generate node: find position in generate_ranks
                for i, rank in enumerate(generate_ranks):
                    if rank == self.rank:
                        sync_rank = len(train_ranks) + i
                        break

            if sync_rank >= 0:
                from ray.util.collective import init_collective_group

                init_collective_group(
                    group_name=sync_group_name,
                    world_size=sync_size,
                    rank=sync_rank,
                    backend="nccl",
                )
                self.train_generate_sync_group = sync_group_name
                enhanced_print(
                    "param_update",
                    None,
                    f"Setup train-generate sync group: rank={sync_rank}, size={sync_size}, group={sync_group_name}",
                )

            return True

        except Exception as e:
            enhanced_print("param_update", None, f"Failed to setup train-generate sync group: {e}")
            return False

    def register_actor_clusters(self, train_ranks: list[int], generate_ranks: list[int], world_size: int):
        """Register train and generate clusters"""
        self.train_ranks = train_ranks
        self.generate_ranks = generate_ranks
        self.world_size = world_size

        enhanced_print(
            "param_update",
            None,
            f"Registered actor clusters: train_ranks={train_ranks}, generate_ranks={generate_ranks}",
        )

    def get_communication_info(self) -> dict[str, Any]:
        """Get communication information"""
        return {
            "async_mode": self.enable_param_async,
            "train_ranks": getattr(self, "train_ranks", []),
            "generate_ranks": getattr(self, "generate_ranks", []),
            "world_size": getattr(self, "world_size", 0),
        }

    def setup_logp_ref_logp_sync(self, logp_rank, ref_logp_rank, size):
        """Setup parameter synchronization for logp and ref_logp workers"""
        if self.enable_param_async:
            # CPU async mode: no additional communication group setup needed
            enhanced_print(
                "param_update", None, "CPU async mode: no additional communication groups needed for logp/ref_logp"
            )
        else:
            # NCCL GPU sync mode: use Ray Collective
            from ray.util.collective import init_collective_group

            # Create independent communication groups for logp and ref_logp
            logp_group_name = "logp_ref_logp_sync"
            init_collective_group(
                group_name=logp_group_name,
                world_size=size,
                rank=logp_rank,
                backend="nccl",
            )
            self.logp_ray_col_name = logp_group_name

            # Create independent communication group for ref_logp
            ref_logp_group_name = "ref_logp_sync"
            init_collective_group(
                group_name=ref_logp_group_name,
                world_size=size,
                rank=ref_logp_rank,
                backend="nccl",
            )
            self.ref_logp_ray_col_name = ref_logp_group_name

            enhanced_print(
                "param_update",
                None,
                f"Setup logp/ref_logp sync: logp_group={logp_group_name}, ref_logp_group={ref_logp_group_name}",
            )

    def sync_param_update(self, func_call=None):
        """GPU sync parameter update - avoid CPU OOM by using GPU memory and NCCL"""
        start_time = time.time()
        enhanced_print("param_update", None, "Starting GPU sync parameter update")

        # Set completion flag to False at the start
        self._param_update_completed = False

        self._sync_meta_info_to_all_nodes()

        # Get parameter groups for bucket processing
        if not self._param_groups:
            groups, group_tensor_count = self._group_tensors_by_metas()
            # Convert groups list to dictionary with bucket names
            self._param_groups = {}
            for i, group in enumerate(groups):
                bucket_name = f"bucket_{i}"
                self._param_groups[bucket_name] = group

        if self.is_train_node():
            # Create generator once and iterate through it efficiently
            per_tensor_param = self.get_params_iter(self.target_device, use_bucketed=False)
        else:
            per_tensor_param = None

        # Process each bucket
        bucket_start_time = time.time()
        total_buckets = len(self._param_groups.items())

        for bucket_idx, (bucket_name, group) in enumerate(self._param_groups.items()):
            bucket_iter_start = time.time()
            enhanced_print(
                "param_update",
                None,
                f"Processing {bucket_name} ({bucket_idx + 1}/{total_buckets}) - {len(group)} tensors",
            )

            # All ranks participate in broadcast
            clear_cache = bucket_idx == total_buckets - 1
            self._sync_broadcast_bucket(per_tensor_param, group, bucket_name, func_call, clear_cache=clear_cache)

            bucket_iter_time = time.time() - bucket_iter_start
            enhanced_print("param_update", None, f"Bucket {bucket_name} completed in {bucket_iter_time:.4f}s")

        total_bucket_time = time.time() - bucket_start_time
        enhanced_print(
            "param_update",
            None,
            f"All {total_buckets} buckets processed in {total_bucket_time:.4f}s "
            f"(avg: {total_bucket_time / total_buckets:.4f}s per bucket)",
        )

        end_time = time.time()
        enhanced_print(
            "param_update", None, f"GPU sync parameter update completed in {end_time - start_time:.2f} seconds"
        )

        # Final memory cleanup to prevent OOM in subsequent operations
        import torch

        enhanced_print("param_update", None, "Starting final memory cleanup...")

        if torch.cuda.is_available():
            # Get memory info before cleanup
            memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
            enhanced_print("param_update", None, f"  Memory before cleanup: {memory_before:.2f} GB")

            # Multiple rounds of cleanup per iteration
            self._force_memory_cleanup(rounds=1)

            # Final memory info
            memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_freed_total = memory_before - memory_after
            enhanced_print(
                "param_update",
                None,
                f"  Final memory: {memory_after:.2f} GB (total freed: {memory_freed_total:.2f} GB)",
            )

        # Set completion flag to True at the end
        self._param_update_completed = True
        return True

    def _force_memory_cleanup(self, rounds=1, verbose=False):
        """Force memory cleanup with multiple rounds"""
        import torch

        for round_num in range(rounds):
            gc.collect()

            if torch.cuda.is_available():
                # CUDA cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()

                if verbose:
                    memory_current = torch.cuda.memory_allocated() / 1024**3  # GB
                    enhanced_print(
                        "param_update", None, f"  Cleanup round {round_num + 1}/{rounds}: {memory_current:.2f} GB"
                    )

    def _sync_broadcast_bucket(self, per_tensor_param, group, bucket_name, func_call, clear_cache=False):
        """All ranks participate in broadcast - optimized with tensor concatenation"""
        from ray.util.collective import broadcast

        def local_clear_cache():
            if not clear_cache:
                return
            self._force_memory_cleanup(rounds=1)

        if self.is_train_node():
            # All train nodes: collect and concatenate tensors
            tensors = []
            tensor_names = []
            tensor_shapes = []
            tensor_dtypes = []

            # Create a set of names we need for this bucket
            needed_names = {meta["name"] for meta in group}

            # Collect only needed tensors from cache
            tensor_count = 0

            # Monitor memory before bucket caching
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3  # GB

            # Iterate through generator to find tensors for current bucket only
            for param_name, tensor in per_tensor_param:
                if param_name in needed_names:
                    tensor_count += 1

                    # Ensure tensor is on GPU
                    if tensor.device.type != "cuda":
                        tensor = tensor.cuda()

                    # Choose whether to flatten based on configuration
                    if self.enable_fused_communication:
                        flattened_tensor = tensor.flatten()
                        tensors.append(flattened_tensor)
                    else:
                        # Use original tensor without flattening
                        tensors.append(tensor)

                    tensor_names.append(param_name)
                    tensor_shapes.append(tensor.shape)
                    tensor_dtypes.append(tensor.dtype)

                # Break early if we found all needed tensors for this bucket
                if len(tensor_names) >= len(needed_names):
                    break

            # Monitor memory after bucket caching
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_delta = memory_after - memory_before
                enhanced_print(
                    "param_update",
                    None,
                    f"  Bucket {bucket_name} memory: {memory_before:.2f}GB -> "
                    f"{memory_after:.2f}GB (delta: +{memory_delta:.2f}GB)",
                )

            # Check if we found all needed tensors
            found_names = set(tensor_names)
            missing_names = needed_names - found_names
            if missing_names:
                enhanced_print("param_update", None, f"ERROR: missing tensors {missing_names} for {bucket_name}")
                return False

            # Check if all tensors have the same dtype
            unique_dtypes = set(tensor_dtypes)

            # Choose communication strategy based on configuration
            if self.enable_fused_communication:
                # Fused communication: concatenate and broadcast once
                if len(unique_dtypes) == 1:
                    # All tensors have the same dtype, can concatenate directly
                    concatenated_tensor = torch.cat(tensors, dim=0)
                else:
                    # Different dtypes detected - convert to common dtype
                    first_dtype = tensors[0].dtype
                    converted_tensors = []
                    for i, t in enumerate(tensors):
                        if t.dtype != first_dtype:
                            converted_t = t.to(first_dtype)
                            converted_tensors.append(converted_t)
                            del t
                        else:
                            converted_tensors.append(t)

                    concatenated_tensor = torch.cat(converted_tensors, dim=0)
                    del converted_tensors

                # Broadcast the concatenated tensor
                broadcast(
                    concatenated_tensor,
                    src_rank=0,
                    group_name=getattr(self, "train_generate_sync_group", self.ray_col_name),
                )

                # Clean up concatenated tensor
                del concatenated_tensor
            else:
                # Individual communication: broadcast each tensor separately
                for i, tensor in enumerate(tensors):
                    broadcast(
                        tensor, src_rank=0, group_name=getattr(self, "train_generate_sync_group", self.ray_col_name)
                    )

            if self.verbose_logging:
                enhanced_print("param_update", None, f"Broadcasted {len(tensors)} individual tensors in {bucket_name}")

            # Clear memory after broadcast
            for tensor in tensors:
                del tensor
            del tensors

            # Force garbage collection and clear cache
            local_clear_cache()
        else:
            # Generation nodes: receive tensors based on communication strategy
            unique_dtypes = set(meta["dtype"] for meta in group)

            if self.enable_fused_communication:
                # Fused communication: receive concatenated tensor and split
                total_size = 0
                for meta in group:
                    shape = meta["shape"]
                    total_size += torch.prod(torch.tensor(shape)).item()

                first_dtype = group[0]["dtype"]
                concatenated_tensor = torch.zeros(total_size, dtype=first_dtype, device="cuda")
                if len(unique_dtypes) != 1:
                    enhanced_print(
                        "param_update",
                        None,
                        f"Warning: mixed dtypes {unique_dtypes} in {bucket_name}, using {first_dtype}",
                    )

                # Receive the concatenated tensor
                broadcast(
                    concatenated_tensor,
                    src_rank=0,
                    group_name=getattr(self, "train_generate_sync_group", self.ray_col_name),
                )

                # Split the concatenated tensor back into individual tensors
                received_tensors = []
                start_idx = 0
                for i, meta in enumerate(group):
                    name = meta["name"]
                    shape = meta["shape"]
                    dtype = meta["dtype"]

                    tensor_size = torch.prod(torch.tensor(shape)).item()
                    tensor_data = concatenated_tensor[start_idx : start_idx + tensor_size]
                    tensor = tensor_data.reshape(shape).to(dtype)
                    received_tensors.append((name, tensor))
                    start_idx += tensor_size

                # Clean up concatenated tensor
                del concatenated_tensor
            else:
                # Individual communication: receive each tensor separately
                # Receive each tensor individually
                received_tensors = []
                for i, meta in enumerate(group):
                    name = meta["name"]
                    shape = meta["shape"]
                    dtype = meta["dtype"]

                    # Allocate tensor for receiving
                    tensor = torch.zeros(shape, dtype=dtype, device="cuda")

                    # Receive the tensor
                    broadcast(
                        tensor, src_rank=0, group_name=getattr(self, "train_generate_sync_group", self.ray_col_name)
                    )

                    received_tensors.append((name, tensor))

            if self.verbose_logging:
                enhanced_print(
                    "param_update", None, f"Received and split {len(received_tensors)} tensors in {bucket_name}"
                )

        # Load weights immediately to avoid memory accumulation (only for generation nodes)
        if self.is_generation_node() and func_call and received_tensors:
            func_call(received_tensors)
            enhanced_print("param_update", None, f"Loaded weights for {bucket_name}")

            # More aggressive memory cleanup
            for name, tensor in received_tensors:
                del tensor
            del received_tensors

            # Force garbage collection but don't clear cache during NCCL communication
            local_clear_cache()

        return True

    def _sync_send_bucket(self, group, bucket_name):
        """Train master: gather all params and broadcast to all nodes"""
        try:
            # Get parameter tensors for this bucket
            bucket_tensors = []
            for meta in group:
                name = meta["name"]
                # Get tensor from model params
                tensor = self._get_tensor_by_name(name)
                if tensor is not None:
                    # Ensure tensor is on GPU
                    if tensor.device.type != "cuda":
                        tensor = tensor.cuda()
                    bucket_tensors.append((name, tensor))

            if not bucket_tensors:
                enhanced_print("param_update", None, f"No tensors found for {bucket_name}")
                return True

            # Use Ray Collective to broadcast to all nodes
            from ray.util.collective import broadcast

            # Broadcast each tensor in the bucket
            for name, tensor in bucket_tensors:
                broadcast(tensor, src_rank=0, group_name=getattr(self, "train_generate_sync_group", self.ray_col_name))
                if self.verbose_logging:
                    enhanced_print("param_update", None, f"Broadcasted {name} in {bucket_name}")

            return True

        except Exception as e:
            enhanced_print("param_update", None, f"Error in _sync_send_bucket for {bucket_name}: {e}")
            return False

    def _sync_recv_bucket(self, group, bucket_name, func_call):
        """Generation node: receive params and load weights"""
        try:
            # Receive parameter tensors for this bucket
            received_tensors = []
            for meta in group:
                name = meta["name"]
                # Create tensor with same shape and dtype
                tensor = torch.zeros(meta["shape"], dtype=meta["dtype"], device="cuda")

                # Receive tensor from train master
                from ray.util.collective import broadcast

                broadcast(tensor, src_rank=0, group_name=getattr(self, "train_generate_sync_group", self.ray_col_name))

                received_tensors.append((name, tensor))
                if self.verbose_logging:
                    enhanced_print("param_update", None, f"Received {name} in {bucket_name}")

            # Load weights immediately to avoid memory accumulation
            if func_call and received_tensors:
                func_call(received_tensors, version=0, group_tensor_count=len(received_tensors))
                enhanced_print("param_update", None, f"Loaded weights for {bucket_name}")

            return True

        except Exception as e:
            enhanced_print("param_update", None, f"Error in _sync_recv_bucket for {bucket_name}: {e}")
            return False

    def _get_tensor_by_name(self, name):
        """Get tensor by name from model parameters"""
        # Use the existing parameter iterator to get tensor
        per_tensor_param = self.get_params_iter(self.target_device, use_bucketed=False)
        for param_name, tensor in per_tensor_param:
            if param_name == name and tensor is not None:
                return tensor

    def sync_per_tensor_generator(self, func_call=None):
        """Choose parameter synchronization method based on enable_param_async"""
        if self.enable_param_async:
            # Use CPU async method (legacy)
            enhanced_print("param_update", None, "Using CPU async parameter update")
            return self.async_param_update_legacy(func_call)
        else:
            # Use NCCL GPU sync method
            enhanced_print("param_update", None, "Using NCCL GPU sync parameter update")
            return self.sync_param_update(func_call)

    def async_param_update_legacy(self, func_call=None, sync_send=False):
        """Legacy CPU async parameter update - use only when enable_param_async=True"""
        start_time = time.time()
        enhanced_print("param_update", None, "Starting CPU async parameter update")

        # First get groups info, ensure send and recv use same grouping
        # Force refresh to ensure dtype accuracy
        if not self._params_meta:
            self.get_params_meta()

        # If no func_call provided, use default update_buffer_data_only method
        if func_call is None:
            # Need to get correct func_call from external, temporarily use a placeholder
            func_call = lambda named_tensors, version: True

        # Record start time for later statistics
        self._param_update_start_time = start_time

        # Start async send
        self._start_async_send()

        # Wait a bit to ensure send thread starts working
        time.sleep(0.1)

        # Start async receive
        self._start_async_recv(func_call)

        if sync_send:
            self.wait_for_send_complete()

    def async_param_update(self, func_call=None, sync_send=False):
        """Deprecated: Use sync_per_tensor_generator instead"""
        return self.sync_per_tensor_generator(func_call)

    def get_params_iter(self, target_device="cpu", use_bucketed=False):
        """Get an iterator for the current parameters."""
        # Check if model_params exists (only train nodes have)
        if self.model_params is None:
            enhanced_print(
                "param_update", None, "get_params_iter: model_params is None, this is not a train node, returning None"
            )
            return None

        # Check if bucket optimization is enabled
        bucket_size_mb = self.send_bucket_size_mb

        if use_bucketed:
            # Use bucket granularity optimized version with cache
            enhanced_print(
                "param_update", None, f"Using bucketed per_tensor_generator with bucket size {bucket_size_mb}MB"
            )
            return self._get_params_iter_bucketed(target_device, bucket_size_mb)
        else:
            # Use original version
            enhanced_print("param_update", None, "Using original per_tensor_generator")
            per_tensor_param = per_tensor_generator(
                self.model_params,
                self.model_config,
                self.weight_converter,
                self.transformer_config,
                self.layer_name_mapping,
                target_device=target_device,
            )

        return per_tensor_param

    def _get_params_iter_bucketed(self, target_device, bucket_size_mb):
        """Simplified bucketed parameter iterator - no prefetch, directly return tensor data"""
        # Get tensor data
        bucket_generator = per_tensor_generator(
            self.model_params,
            self.model_config,
            self.weight_converter,
            self.transformer_config,
            self.layer_name_mapping,
            target_device=target_device,
        )

        # Directly return tensor data, no prefetch
        return bucket_generator

    def _get_params_iter_bucketed_enhanced(self, target_device, bucket_size_mb):
        """Enhanced version: get bucketed parameter iterator, including params_meta and global mapping"""
        # Get parameter metadata
        if not self._params_meta:
            self.get_params_meta()

        # Check if params_meta is empty
        if not self._params_meta:
            enhanced_print("param_update", None, "Warning: params_meta is empty, returning empty results")
            return {}, [], 0

        # Use simplified grouping strategy
        groups, group_tensor_count = self._group_tensors_by_metas()

        # Check if groups is empty
        if not groups:
            enhanced_print("param_update", None, "Warning: groups is empty, returning empty results")
            return {}, [], 0

        # Create global name to tensor mapping
        global_tensor_map = {}

        if self.model_params is None:
            # Model params empty, return empty result, only need groups info
            return global_tensor_map, groups, group_tensor_count

        # Get tensor data
        per_tensor_param = self._get_params_iter_bucketed(target_device, bucket_size_mb)

        # Build global mapping
        for name, tensor in per_tensor_param:
            if tensor is not None and self.is_train_master_node():
                global_tensor_map[name] = tensor.to(target_device)

        if self.is_train_master_node():
            enhanced_print("param_update", None, f"Created global tensor map with {len(global_tensor_map)} tensors")

        # Return enhanced iterator with meta info and global mapping
        return global_tensor_map, groups, group_tensor_count

    def clear_gpu_cache(self):
        """Clear GPU cache, release GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            enhanced_print("param_update", None, "Cleared GPU cache")

    def clear_all_cache(self):
        """Clear all caches, including bucket cache and GPU cache"""
        self.clear_bucketed_cache()
        self.clear_gpu_cache()
        enhanced_print("param_update", None, "Cleared all caches")

    def get_bucketed_cache_info(self):
        """Get bucket cache information"""
        if hasattr(self, "_bucketed_cache"):
            cache_info = {}
            for key, buckets in self._bucketed_cache.items():
                cache_info[key] = {
                    "bucket_count": len(buckets),
                    "total_tensors": sum(len(bucket) for bucket in buckets),
                }
            return cache_info
        return {}

    def get_params_meta(self, force_refresh=False):
        """Get parameter metadata"""
        # Check if _params_meta is initialized and not empty, and not forcing refresh
        if not force_refresh and hasattr(self, "_params_meta") and self._params_meta and len(self._params_meta) > 0:
            return self._params_meta

        # Check if model_params is available
        if self.model_params is None:
            enhanced_print("param_update", None, "Warning: model_params is None, cannot get params meta")
            return []

        per_tensor_param = self._get_params_iter_bucketed(self.target_device, self.send_bucket_size_mb)

        if per_tensor_param is None:
            enhanced_print("param_update", None, "Error: per_tensor_param returned None")
            return []

        # Always regenerate meta-info to ensure dtype accuracy
        self._params_meta = []

        for key, tensor in per_tensor_param:
            if tensor is not None:
                meta = {
                    "name": key,
                    "shape": tensor.shape,
                    "dtype": tensor.dtype,
                    "size": tensor.numel() * tensor.element_size(),
                }
                self._params_meta.append(meta)
            else:
                # Handle None tensor
                meta = {
                    "name": key,
                    "shape": (),
                    "dtype": torch.float32,
                    "size": 0,
                }
                self._params_meta.append(meta)

        enhanced_print(
            "param_update",
            None,
            f"Generated {len(self._params_meta)} parameter metadata entries using bucketed generator",
        )

        return self._params_meta

    def set_params_meta(self, params_meta):
        """Set the parameters metadata."""
        self._params_meta = params_meta

    def is_meta_info_needs_sync(self):
        """Check if meta-info has been updated and needs sync"""
        return getattr(self, "_meta_info_needs_sync", True)

    def mark_meta_info_synced(self):
        """Mark that meta-info has been synced"""
        self._meta_info_needs_sync = False

    def _sync_meta_info_to_all_nodes(self):
        """Sync updated meta-info to all nodes using Ray communication"""
        # Skip if meta-info has already been synced
        if not self.is_meta_info_needs_sync():
            enhanced_print("param_update", None, "Meta-info already synced, skipping broadcast")
            return

        # All nodes participate in broadcast - train master sends, others receive
        from ray.util.collective import broadcast

        if self.is_train_node():
            self.get_params_meta(force_refresh=True)
        if self.is_train_master_node():
            # Train master node: prepare and send meta-info
            meta_info_tensor = self._serialize_meta_info()
            enhanced_print(
                "param_update",
                None,
                f"Train master: preparing to broadcast meta-info: {len(self._params_meta)} entries",
            )
        else:
            # All other nodes: prepare to receive meta-info
            # First broadcast the size, then broadcast the actual data
            size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
            broadcast(size_tensor, src_rank=0, group_name=getattr(self, "train_generate_sync_group", self.ray_col_name))

            # Create tensor with the received size
            meta_info_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device="cuda")

        # All nodes participate in broadcast (collective operation)
        broadcast(
            meta_info_tensor, src_rank=0, group_name=getattr(self, "train_generate_sync_group", self.ray_col_name)
        )

        if not self.is_train_master_node():
            # Non-master nodes: deserialize received meta-info
            self._params_meta = self._deserialize_meta_info(meta_info_tensor)
            enhanced_print(
                "param_update", None, f"Received meta-info from train master: {len(self._params_meta)} entries"
            )
        else:
            # Train master: broadcast completed
            enhanced_print(
                "param_update", None, f"Broadcasted meta-info to all nodes: {len(self._params_meta)} entries"
            )

        self.mark_meta_info_synced()

    def _serialize_meta_info(self):
        """Serialize meta-info to a tensor for broadcasting"""
        import pickle

        # Serialize meta-info to bytes
        meta_info_bytes = pickle.dumps(self._params_meta)

        # Convert to tensor
        meta_info_tensor = torch.frombuffer(meta_info_bytes, dtype=torch.uint8)

        # First broadcast the size
        size_tensor = torch.tensor([len(meta_info_tensor)], dtype=torch.int64, device="cuda")
        from ray.util.collective import broadcast

        broadcast(size_tensor, src_rank=0, group_name=getattr(self, "train_generate_sync_group", self.ray_col_name))

        # Return the actual data tensor
        return meta_info_tensor.cuda()

    def _deserialize_meta_info(self, meta_info_tensor):
        """Deserialize meta-info from a tensor"""
        import pickle

        # Convert tensor to bytes
        meta_info_bytes = meta_info_tensor.cpu().numpy().tobytes()

        # Deserialize
        meta_info = pickle.loads(meta_info_bytes)

        return meta_info

    def preduce_per_tensor_generator(self, convert_generator_to_list=True):
        """Asynchronously get the current parameters."""
        per_tensor_param = self.get_params_iter("cpu", use_bucketed=True)

        if not convert_generator_to_list:
            return per_tensor_param

        # convert the async generator to a list
        param_list = []
        for key, tensor in per_tensor_param:
            param_list.append((key, tensor))
        enhanced_print("param_update", None, f"Total tensors in per_tensor_generator: {len(param_list)}")
        return param_list

    def consume_per_tensor_generator(self, per_tensor_param, func_call):
        """Consume the per_tensor_generator with a function call."""
        # Check if per_tensor_param is None
        if per_tensor_param is None:
            enhanced_print("param_update", None, "Warning: per_tensor_param is None, skipping consumption")
            return

        # Check if event loop exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Directly call coroutine function
            coro = func_call(per_tensor_param)
            loop.run_until_complete(coro)
        except StopIteration:
            # Handle StopIteration exception
            enhanced_print(
                "param_update", None, "Warning: StopIteration in consume_per_tensor_generator, generator may be empty"
            )
        except Exception as e:
            # Handle other exceptions
            enhanced_print("param_update", None, f"Error in consume_per_tensor_generator: {e}")
            import traceback

            traceback.print_exc()
            raise e

    def _group_tensors_by_metas(self):
        """Simplified tensor grouping strategy - use synchronized params_meta to ensure consistency"""
        if not self._params_meta:
            self.get_params_meta()

        # Check if params_meta is empty
        if not self._params_meta:
            enhanced_print("param_update", None, "Warning: params_meta is empty after get_params_meta()")
            return []

        target_bucket_size = self.send_bucket_size_mb * 1024 * 1024

        # Simple grouping strategy, fill bucket in order
        groups = []
        current_group = []
        current_size = 0

        for meta in self._params_meta:
            if meta is None:
                enhanced_print("param_update", None, "Warning: Found None meta in params_meta, skipping")
                continue

            tensor_size = self._calculate_tensor_size(meta)

            # If adding this tensor exceeds target size, create new group
            if current_size + tensor_size > target_bucket_size and current_group:
                groups.append(current_group)
                current_group = [meta]
                current_size = tensor_size
            else:
                current_group.append(meta)
                current_size += tensor_size

        # Add last group
        if current_group:
            groups.append(current_group)

        # Calculate statistics
        group_sizes = []
        group_tensor_count = 0
        for group in groups:
            total_size = sum(self._calculate_tensor_size(meta) for meta in group)
            group_tensor_count += len(group)
            group_sizes.append(total_size)

        return groups, group_tensor_count

    def _calculate_tensor_size(self, meta):
        """Calculate the size of a tensor."""
        return meta["size"]

    def _execute_async_func_call(self, func_call, named_tensors):
        """Execute async func_call"""
        result = func_call(named_tensors)
        return result

    def _async_recv_bucket_with_store_ref(self, store_ref, version, bucket_name, func_call, group_tensor_count):
        """Async recv single bucket - receive data and update buffer"""

        # Check if store_ref is valid
        if not store_ref:
            enhanced_print("param_update", None, f"Async recv: no store_ref provided for {bucket_name}")
            return False

        get_start_time = time.time()

        # [(name, tensor)]
        received_tensors = cross_process_ray_get(store_ref)

        get_time = time.time() - get_start_time

        if not received_tensors:
            enhanced_print("param_update", None, f"Async recv: no data available for version {version}")
            return False

        if self.verbose_logging:
            enhanced_print(
                "param_update",
                None,
                f"Async recv: Ray get {len(received_tensors)} tensors completed "
                f"for {bucket_name} (version {version}) in {get_time:.3f}s",
            )

        named_tensors = received_tensors

        # Update buffer data
        if func_call and named_tensors:
            # Directly call func_call to update buffer data
            buffer_success = func_call(named_tensors, version, group_tensor_count)

            return buffer_success

        return True

    def _start_async_send(self):
        """Start async send thread"""
        if not hasattr(self, "_async_send_thread") or not self._async_send_thread.is_alive():
            self._async_send_thread = threading.Thread(target=self._async_send_worker, daemon=True)
            self._async_send_thread.start()
            # enhanced_print("param_update", None, "Started async send worker thread")

    def _start_async_recv(self, func_call):
        """Start async recv thread"""
        if not hasattr(self, "_async_recv_thread") or not self._async_recv_thread.is_alive():
            self._async_recv_thread = threading.Thread(target=self._async_recv_worker, args=(func_call,), daemon=True)
            self._async_recv_thread.start()
            # enhanced_print("param_update", None, "Started async recv worker thread")

    @torch.no_grad()
    def _async_send_worker(self):
        """Async send worker"""
        enhanced_print("param_update", None, "Async send worker started")

        t1 = time.time()

        # Get parameter metadata
        if not self._params_meta:
            self.get_params_meta()

        # Use enhanced bucketed function to get tensor data and grouping info
        global_tensor_map, groups, group_tensor_count = self._get_params_iter_bucketed_enhanced(
            self.target_device, bucket_size_mb=self.send_bucket_size_mb
        )

        if not self.is_train_master_node():
            # Non-send nodes can exit directly after syncing params
            return

        if self.is_train_master_node() and (not global_tensor_map or not groups):
            enhanced_print("param_update", None, "Async send: no tensor data or groups available")
            return

        # Use unified version (each async send advances global version once)
        version = self.current_version + 1
        self.current_version = version

        enhanced_print(
            "param_update",
            None,
            f"Async send: processing {len(groups)} buckets (version:{version}) with {len(global_tensor_map)} tensors",
        )

        success_count = 0
        for i, group in enumerate(groups):
            bucket_name = f"bucket_{i}"

            bucket_tensors = []
            found_count = 0
            for meta in group:
                name = meta["name"]
                if name in global_tensor_map:
                    tensor = global_tensor_map[name]
                    bucket_tensors.append((name, tensor))
                    found_count += 1

            if not bucket_tensors:
                enhanced_print("param_update", None, f"Async send: no tensors found for {bucket_name}")
                continue

            success = self._async_send_bucket(bucket_tensors, 0, bucket_name, version=version)
            if success:
                success_count += 1
                # enhanced_print("param_update", None, f"Async send: completed {bucket_name} (version {version})")
            else:
                enhanced_print("param_update", None, f"Async send: failed {bucket_name} (version {version})")

            if self.verbose_logging:
                enhanced_print(
                    "param_update", None, f"Async send: {bucket_name} - {found_count}/{len(group)} tensors sent"
                )

        t2 = time.time()
        enhanced_print(
            "param_update",
            None,
            f"Async send: completed {success_count}/{len(groups)} buckets, "
            f"cost time:{t2 - t1:.2f}, worker thread ending",
        )

    @torch.no_grad()
    def _async_recv_worker(self, func_call):
        """Async recv worker"""
        t1 = time.time()

        if not self.is_generation_master_node():
            # Non-generator master node, no receive operation
            return True

        enhanced_print("param_update", None, "Async recv worker started")

        # Get parameter metadata
        if not self._params_meta:
            self.get_params_meta()

        # Use enhanced bucketed function to get grouping info
        _, groups, group_tensor_count = self._get_params_iter_bucketed_enhanced(
            self.target_device, bucket_size_mb=self.send_bucket_size_mb
        )

        if not groups:
            enhanced_print("param_update", None, "Async recv: no groups available")
            return

        # Initialize current version of receiver (will use version from queue)
        version = self.current_version
        thread_id = threading.get_ident()

        enhanced_print(
            "param_update", None, f"Async recv: processing {len(groups)} buckets ({group_tensor_count} tensors)"
        )
        # Process each bucket, receive in order - ensure same order as send
        success_count = 0
        for i, group in enumerate(groups):
            bucket_name = f"bucket_{i}"

            # Get object_refs from store_refs_queue
            store_ref = None
            if self.store_refs_queue is not None:
                # Get object_refs info from queue, use blocking mode
                queue_data = self.store_refs_queue[self.engine_idx()].get()
                if self.verbose_logging:
                    enhanced_print(
                        "param_update",
                        None,
                        f"Async recv: got queue_data for "
                        f"{queue_data.get('bucket_name') if queue_data else 'None'}, "
                        f"expecting {bucket_name}",
                    )

                # Verify bucket_name order consistency
                expected_bucket_name = bucket_name
                actual_bucket_name = queue_data.get("bucket_name") if queue_data else None

                if actual_bucket_name != expected_bucket_name:
                    raise AssertionError(
                        f"ERROR: i:{i}, thread id: {thread_id}, rank:{self.rank}, "
                        f"engine_idx:{self.engine_idx()} Bucket order mismatch! "
                        f"Expected {expected_bucket_name}, got {actual_bucket_name}, exit"
                    )

                bucket_name = actual_bucket_name
                store_ref = queue_data.get("object_refs") if queue_data else None
                # Get version number for this bucket from queue
                received_version = queue_data.get("version") if queue_data else None
                if received_version is not None:
                    version = received_version
                if self.verbose_logging:
                    enhanced_print(
                        "param_update",
                        None,
                        f"i:{i}, thread id: {thread_id}, rank:{self.rank}, "
                        f"engine_idx:{self.engine_idx()}, Async recv: got "
                        f"{len(store_ref)} object_refs for {bucket_name} "
                        f"(version {version})",
                    )

            # Use _async_recv_bucket_with_store_ref to receive data
            received_tensors = self._async_recv_bucket_with_store_ref(
                store_ref, version, bucket_name, func_call, group_tensor_count
            )

            if received_tensors:
                success_count += 1
                # enhanced_print("param_update", None, f"i:{i}, rank:{self.rank} "
                # f"Async recv: completed {bucket_name} (version {version})")
            else:
                enhanced_print(
                    "param_update",
                    None,
                    f"i:{i}, rank:{self.rank} Async recv: failed {bucket_name} (version {version})",
                )

        t2 = time.time()
        # send / recv run independently, version from send
        self.current_version = version
        enhanced_print(
            "param_update",
            None,
            f"Async recv: completed {success_count}/{len(groups)} "
            f"buckets(version:{version}), cost time:{t2 - t1:.2f}, worker thread ending",
        )

    def _async_send_bucket(self, bucket_tensors, ray_rank, bucket_name, version=None):
        """Async send single bucket - train node (using Ray put/get)"""
        if self.verbose_logging:
            enhanced_print(
                "param_update", None, f"Async send: starting {bucket_name} with {len(bucket_tensors)} tensors"
            )

        # If no version provided, use default value
        if version is None:
            version = 1

        # Check if bucket_tensors is empty
        if not bucket_tensors:
            enhanced_print("param_update", None, f"ERROR: bucket_tensors is empty for {bucket_name}")
            return False

        # Ensure all tensors are on CPU, prepare for Ray put
        cpu_tensors = []
        for name, tensor in bucket_tensors:
            if tensor.device.type == "cuda":
                tensor = tensor.cpu()
            cpu_tensors.append((name, tensor))

        fused_tensors = cpu_tensors

        # Check fusion result
        if not fused_tensors:
            enhanced_print("param_update", None, f"Async send: no fused tensors for {bucket_name}, skipping broadcast")
            return True

        # Send fusion tensors for each dtype group
        def broadcast_tensor(tensor, src_rank, group_name):
            # Use Ray put/get async communication, tensor is already on CPU
            # Check if tensor is valid
            if tensor is None:
                enhanced_print("param_update", None, f"Invalid tensor for {group_name}, skipping broadcast")
                return None

            # Use cross_process_ray_put for put operation
            start_time = time.time()
            object_ref = cross_process_ray_put(tensor, version=version)  # Use version passed in
            put_time = time.time() - start_time

            if self.verbose_logging:
                enhanced_print("param_update", None, f"Async send: {group_name} - put={put_time * 1000:.1f}ms")

            return object_ref

        # Send fusion tensors
        object_refs = []  # Store object_refs for cross-ray-actor communication

        # Broadcast the whole [(name, tensor)]
        object_refs = broadcast_tensor(fused_tensors, ray_rank, bucket_name)

        # Store object_refs to store_refs_queue, for recv thread to use
        if self.store_refs_queue is not None and object_refs:
            for queue_idx in range(len(self.store_refs_queue)):
                self.store_refs_queue[queue_idx].put(
                    {"bucket_name": bucket_name, "version": version, "object_refs": object_refs}
                )

        if self.verbose_logging:
            enhanced_print("param_update", None, f"Async send: {bucket_name} sent successfully")

        return True

    def wait_for_send_complete(self):
        """Block wait for send to complete"""
        if not self.enable_param_async:
            # NCCL GPU sync mode: no async threads, just add barrier
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        # CPU async mode: wait for async send thread
        if hasattr(self, "_async_send_thread") and self._async_send_thread.is_alive():
            self._async_send_thread.join()

        # Add distributed barrier to ensure all nodes complete send before proceeding
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            enhanced_print("param_update", None, "All nodes completed send phase")

    def wait_for_recv_complete(self):
        """Block wait for recv to complete"""
        if not self.enable_param_async:
            # NCCL GPU sync mode: wait for parameter update completion flag
            while not self._param_update_completed:
                time.sleep(0.1)  # Small sleep to avoid busy waiting
            enhanced_print("param_update", None, "NCCL sync mode: parameter update completed")
            return
        else:
            # CPU async mode: wait for async recv thread
            if hasattr(self, "_async_recv_thread") and self._async_recv_thread.is_alive():
                self._async_recv_thread.join()

        # Add distributed barrier to ensure all nodes complete recv before proceeding
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            enhanced_print("param_update", None, "All nodes completed recv phase")
