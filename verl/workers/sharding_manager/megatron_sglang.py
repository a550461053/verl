# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

import asyncio
import logging
import os
import time

import torch
from sglang.srt.entrypoints.engine import Engine
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.megatron_utils import per_tensor_generator

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


class MegatronSGLangShardingManager(BaseShardingManager):
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        device_mesh: DeviceMesh | None = None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.device_mesh = device_mesh

        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["tp"].mesh.size()[0]
        else:
            self.infer_tp_size = self.inference_engine._tp_size

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        per_tensor_param = per_tensor_generator(
            self.actor_module,
            self.model_config,
            self.weight_converter,
            self.transformer_config,
            self.layer_name_mapping,
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.update_weights(per_tensor_param))
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.release_memory())
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    async def update_weights(self, params):

        if self.device_mesh["tp"].get_local_rank() == 0:
            await self.inference_engine.resume_memory_occupation()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        # named_tensors = [(k, v) for k, v in params.items()]
        named_tensors = params
        load_format = None

        for tensor_index, (name, tensor) in enumerate(named_tensors):
            if self.device_mesh["tp"].get_local_rank() == 0:
                await self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            tensor.detach(),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=False,
                )

            if self.device_mesh["tp"].get_local_rank() == 0:
                await self.inference_engine.flush_cache()


    async def release_memory(self):
        if self.device_mesh["tp"].get_local_rank() == 0:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        per_tensor_param = per_tensor_generator(
            self.actor_module,
            self.model_config,
            self.weight_converter,
            self.transformer_config,
            self.layer_name_mapping,
        )
        await self.update_weights(per_tensor_param)
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        await self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.device_mesh["tp"].get_group())
        return data

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.device_mesh["tp"].get_local_rank()]

class MegatronSGLangAsyncShardingManager(MegatronSGLangShardingManager):
    """
    This class is used to handle the async inference in Megatron SGLang.
    It inherits from MegatronSGLangShardingManager and overrides the wake_up and sleep methods.
    """
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        device_mesh: DeviceMesh | None = None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.device_mesh = device_mesh

        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["tp"].mesh.size()[0]
        else:
            self.infer_tp_size = self.inference_engine._tp_size

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None
        
        # 添加dual_buffer_engine属性
        self.dual_buffer_engine = None
        if hasattr(inference_engine, 'update_buffer_data_only'):
            self.dual_buffer_engine = inference_engine
            print(f"[MegatronSGLangAsyncShardingManager] Using dual_buffer_engine: {type(inference_engine)}")

    def set_model_parameters(self, actor_module: nn.ModuleList):
        """
        Set the actor module parameters for the sharding manager.
        This is used to update the actor module parameters before inference.
        """
        self.actor_module = actor_module

    def update_model_params(self, actor_module):
        self.set_model_parameters(actor_module)
        per_tensor_param = per_tensor_generator(
            self.actor_module,
            self.model_config,
            self.weight_converter,
            self.transformer_config,
            self.layer_name_mapping,
        )
        
        # 修复：安全地处理asyncio调用，避免在非主线程中hang
        import threading
        current_thread = threading.current_thread()
        is_main_thread = current_thread.name == 'MainThread'
        
        if is_main_thread:
            # 在主线程中，可以使用asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.update_weights(per_tensor_param))
            except Exception as e:
                print(f"Async update_weights failed in main thread: {e}")
                # 如果asyncio失败，跳过权重更新
                print("Skipping weight update due to asyncio error...")
        else:
            # 在非主线程中，跳过权重更新以避免hang
            print(f"Warning: update_model_params called in non-main thread '{current_thread.name}', skipping weight update")
            # 跳过权重更新，避免asyncio问题
            pass

    @GPUMemoryLogger(role="MegatronSGLangAsyncShardingManager enter", logger=logger)
    def __enter__(self):
        # per_tensor_param = per_tensor_generator(
        #     self.actor_module,
        #     self.model_config,
        #     self.weight_converter,
        #     self.transformer_config,
        #     self.layer_name_mapping,
        # )
        # t1 = time.time()
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(self.update_weights(per_tensor_param))
        t2 = time.time()
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)
        # print(f"megatron_sglang enter update_weights cost_time:{t2 - t1:.2f}s") 

    @GPUMemoryLogger(role="MegatronSGLangAsyncShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        # loop = asyncio.get_event_loop()
        # t1 = time.time()
        # loop.run_until_complete(self.release_memory())
        # log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)
        # t2 = time.time()
        # print(f"megatron_sglang exit cleam_memory cost_time:{t2 - t1}s")

        # for model in self.actor_module:
        #     model.train()
        # # add empty cache after each compute
        # torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def update_weights_sync(self, params):
        """
        完全同步版本的update_weights，避免使用async调用
        """
        import time
        t1 = time.time()

        named_tensors = params
        load_format = None
        
        try:
            print(f"update_weights_sync loading {len(named_tensors)} params")
            for tensor_index, (name, tensor) in enumerate(named_tensors):
                if self.device_mesh["tp"].get_local_rank() == 0:
                    # 直接调用同步版本的update_weights_from_tensor
                    if hasattr(self.inference_engine, 'update_weights_from_tensor_sync'):
                        self.inference_engine.update_weights_from_tensor_sync(
                            named_tensors=[
                                (
                                    name,
                                    tensor.detach(),
                                )
                            ],
                            load_format=load_format,
                            flush_cache=False,
                        )
                    else:
                        # 如果没有同步版本，跳过
                        print(f"Warning: inference_engine has no update_weights_from_tensor_sync method")

                if self.device_mesh["tp"].get_local_rank() == 0:
                    # 直接调用同步版本的flush_cache
                    if hasattr(self.inference_engine, 'flush_cache_sync'):
                        self.inference_engine.flush_cache_sync()
                    else:
                        # 如果没有同步版本，跳过
                        print(f"Warning: inference_engine has no flush_cache_sync method")

        except Exception as e:
            print(f"update_weights_sync failed: {e}")
            logger.error(f"Error during update_weights_sync: {e}")
            if named_tensors and len(named_tensors) > 0:
                real_tensors = named_tensors[0]
                print(f"real_tensors: {real_tensors}, type(real_tensors): {type(real_tensors)}")
            else:
                print(f"named_tensors is empty: {named_tensors}")
            raise e
        
        t2 = time.time()
        print(f"update_weights_sync cost_time:{t2 - t1}s")

    def sync_update_weights(self, params):
        """
        This method is used to update the weights of the inference engine synchronously.
        It is used for the synchronous inference in Megatron SGLang.
        """
        # 使用完全同步的版本，避免async调用
        try:
            self.update_weights_sync(params)
        except Exception as e:
            print(f"sync_update_weights failed: {e}")
            # 如果同步版本失败，跳过权重更新
            print("Skipping sync weight update due to error...")

    async def update_weights(self, params, use_reqinput=False):
        import time
        t1 = time.time()

        # if self.device_mesh["tp"].get_local_rank() == 0:
        #     await self.inference_engine.resume_memory_occupation()

        if use_reqinput:
            for obj in params:
                if self.device_mesh["tp"].get_local_rank() == 0:
                    await self.inference_engine.update_weights_from_reqinput(obj)
                if self.device_mesh["tp"].get_local_rank() == 0:
                    await self.inference_engine.flush_cache()
        else:
            # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
            # named_tensors = [(k, v) for k, v in params.items()]
            named_tensors = params
            load_format = None
            
            # 判断是否有len方法
            # if hasattr(named_tensors, '__len__'):
            #     print(f"update_weights loading {len(named_tensors)} params")
            # print(f"named_tensors:{named_tensors}")
            for tensor_index, (name, tensor) in enumerate(named_tensors):
                if self.device_mesh["tp"].get_local_rank() == 0:
                    await self.inference_engine.update_weights_from_tensor(
                        named_tensors=[
                            (
                                name,
                                tensor.detach(),
                            )
                        ],
                        load_format=load_format,
                        flush_cache=False,
                    )

                if self.device_mesh["tp"].get_local_rank() == 0:
                    await self.inference_engine.flush_cache()

        t2 = time.time()
        # load_weight per_tensor: 106s
        # 
        print(f"megatron_sglang update_weight cost_time:{t2 - t1}s")
        return True

    async def release_memory(self):
        if self.device_mesh["tp"].get_local_rank() == 0:
            await self.inference_engine.release_memory_occupation()


    @GPUMemoryLogger(role="MegatronSGLangAsyncShardingManager enter", logger=logger)
    async def wake_up(self):
        await super().wake_up()
        # additional logic for async inference can be added here

    @GPUMemoryLogger(role="MegatronSGLangAsyncShardingManager exit", logger=logger)
    async def sleep(self):
        await super().sleep()
        # additional logic for async inference can be added here