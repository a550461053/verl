import json
import os
import uuid
import asyncio
import atexit
import concurrent.futures
from collections import OrderedDict
import sys
import torch
import time
import ray
from ray.util.queue import Queue
import numpy as np
from copy import deepcopy
from pprint import pprint
from tqdm import tqdm

from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayClassWithInitArgs,
    AsyncLLMServerManager,
    Role,
    OmegaConf,
    create_colocated_worker_cls,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    _timer,
    compute_advantage,
    compute_response_mask,
    agg_loss,
    AdvantageEstimator,
    DataProto,
    apply_kl_penalty,
)
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage

# 从新的 pipeline 模块导入
from verl.trainer.ppo.pipeline import (
    AsyncPipeline,
    enhanced_print,
    PIPELINE_END_SIGNAL,
    PIPELINE_START_SINGLE,
    ROLE_COLORS,
)


class RayPPOAsyncPipelineTrainer(RayPPOTrainer):
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the RayPPOAsyncPipelineTrainer.

        Args:
            config: Configuration for the trainer.
            device_name: Name of the device to use.
            use_critic: Whether to use a critic in training.
            use_reference_policy: Whether to use a reference policy.
            ref_in_actor: Whether the reference policy is integrated into the actor.
            use_rm: Whether to use a reward model.
            hybrid_engine: Whether to use a hybrid engine for actor and rollout.
            ray_worker_group_cls: Custom Ray worker group class.
        """
        super().__init__(*args, **kwargs)
        
        self._overlap_param_update = self.config.actor_rollout_ref.get("overlap_param_update", True)
        self._async_logp_ref_logp = self.config.actor_rollout_ref.get("async_logp_ref_logp", True)
        self._async_pipeline = AsyncPipeline(max_queue_size=self.config.actor_rollout_ref.rollout.get("max_queue_size", 2))        
        if not self._async_logp_ref_logp:
            print(f"roles in async pipeline: {self._async_pipeline.role}", flush=True)
            self._async_pipeline.role.remove("logp")
            self._async_pipeline.role.remove("ref_logp")
        
        self.global_steps = 0
        self.dataloader_global_step = 0
        self.generate_global_step = 0


    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        t1 = time.time()
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            # raise NotImplementedError
            for role, role_name in [(Role.Actor, "actor"), (Role.Rollout, "rollout")]:
                resource_pool = self.resource_pool_manager.get_resource_pool(role)
                worker_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[role], config=self.config.actor_rollout_ref, role=role_name)
                self.resource_pool_to_cls[resource_pool][role_name] = worker_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # breakpoint()
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        t2 = time.time()
        print(f"===== finished creating resource pools and worker classes in {t2 - t1:.2f} seconds =====", flush=True)

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        # breakpoint()
        
        async_pipline_init = self.config.actor_rollout_ref.get("async_pipeline_init", True)
        
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        atexit.register(_executor.shutdown, wait=True)
        _async_tasks = []

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            t1 = time.time()
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            t2 = time.time()
            print(f"using resource pool {resource_pool} cost time:{t2 - t1:.2}s", flush=True)
        
        t1 = time.time()
        
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            if async_pipline_init:
                _async_tasks.append(_executor.submit(self.ref_policy_wg.init_model))
            else:
                self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        t2 = time.time()

        if "actor_rollout" in all_wg:
            # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
            self.actor_rollout_wg = all_wg["actor_rollout"]
            self.actor_rollout_wg.init_model()
        else:
            # if we are not using hybrid engine, we should create actor and rollout separately
            self.actor_wg = all_wg["actor"]
            self.rollout_wg = all_wg["rollout"]
            
            if async_pipline_init:
                print(f"===== initializing actor worker asynchronously =====", flush=True)
                # Use the executor to run the actor initialization in a separate thread
                _async_tasks.append(_executor.submit(self.actor_wg.init_model))
                # Use the executor to run the rollout initialization in a separate thread
                _async_tasks.append(_executor.submit(self.rollout_wg.init_model))
            else:
                self.actor_wg.init_model()
                self.rollout_wg.init_model()

        # wait for all async tasks to finish
        if len(_async_tasks) > 0:
            print(f"===== waiting for async tasks to finish =====", flush=True)
            for task in _async_tasks:
                task.result()

        if async_pipline_init:
            members = self.actor_wg.workers + self.rollout_wg.workers
            col_size = len(members)
            col_name = "actor_rollout_sync"
            col_ranks = list(range(col_size))
            
            # 根据配置选择通信方式
            comm_type = getattr(self.config.actor_rollout_ref, "comm_type", "ray")  # "ray" or "pytorch"
            backend = getattr(self.config.actor_rollout_ref, "comm_backend", "nccl")    # "nccl" or "gloo"
            
            if comm_type == "ray":
                # 使用Ray Collective（支持NCCL）
                import ray.util.collective as col
                col.create_collective_group(
                    actors=members,
                    world_size=col_size,
                    ranks=col_ranks,
                    backend=backend,
                    group_name=col_name
                )
                print(f"Created Ray Collective {backend.upper()} group: {col_name}")
            elif comm_type == "pytorch":
                # 使用PyTorch distributed，不需要Ray Collective初始化
                print(f"Using PyTorch {backend} backend, skipping Ray Collective initialization")
            else:
                print(f"Unsupported comm_type: {comm_type}")
            
            # 设置worker的通信配置
            actor_len = len(self.actor_wg.workers)
            self.actor_wg.setup_for_ray_col(0, col_size, col_name, backend)
            if actor_len != col_size:
                self.rollout_wg.setup_for_ray_col(actor_len, col_size, col_name, backend)
            
            self.actor_wg.check_for_ray_col(col_name)
            if actor_len != col_size:
                self.rollout_wg.check_for_ray_col(col_name)
            
            # share global queue
            from ray.util.queue import Queue
            engine_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            # TODO: support PP/EP
            engine_nums = (col_size - actor_len) // engine_tp_size
            self._global_queue = [Queue() for _ in range(engine_nums)]
            self.actor_wg.setup_for_queue(self._global_queue)
            self.rollout_wg.setup_for_queue(self._global_queue)

            # sync param_meta
            param_meta = self.actor_wg.get_params_meta()
            self.rollout_wg.set_params_meta(param_meta)
        
        t3 = time.time()
        print(f"===== finished async_pipline:{async_pipline_init} initializing workers in {t3 - t2:.2f},{t2-t1:.2f} seconds =====", flush=True)

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )


    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    async def sync_weight(self, sync_thread=False):
        """同步权重到所有worker"""
        if sync_thread:
            self.actor_wg.sync_per_tensor_generator()
            ray.get(self.rollout_wg.sync_per_tensor_generator())
        else:
            self.actor_wg.sync_per_tensor_generator()
            self.rollout_wg.sync_per_tensor_generator()


    async def rollout(self):
        """执行rollout逻辑"""
        while True:
            # 模拟rollout逻辑
            print("Performing rollout...")
            await asyncio.sleep(1)  # 模拟异步操作

    def _pre_process_batch(self, batch: DataProto):
        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        _gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        return _gen_batch

    def get_next_batch(self):
        for epoch in range(self.config.trainer.total_epochs):
            self.epoch = epoch
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = self._pre_process_batch(batch)
                
                # self.global_steps += 1
                self.dataloader_global_step += 1
                yield self.dataloader_global_step, gen_batch, batch_dict
        yield -1, PIPELINE_END_SIGNAL, PIPELINE_END_SIGNAL  # 用于标识数据流结束
            
    async def dataloader_loop(self):
        
        dataloader_batch_iter = self.get_next_batch()
        pipeline_start = await self._async_pipeline.pull(src_role="train", dst_role="dataloader")
        print(f"[dataloader] loop started with pipeline_start: {pipeline_start}, {pipeline_start == PIPELINE_START_SINGLE}")
        max_pending_size = 2
        while True:
            if pipeline_start == PIPELINE_START_SINGLE:
                
                cur_queue = self._async_pipeline.get_cur_queue(src_role="train", dst_role="rollout")
                
                if cur_queue.qsize() >= max_pending_size:
                    # print(f"Current queue size {cur_queue.qsize()} is greater than or equal to max_pending_size {max_pending_size}. Waiting...")
                    # 等待一段时间后再继续
                    await asyncio.sleep(1)
                    continue
                
                # 获取下一个批次
                # gen_batch = await asyncio.get_event_loop().run_in_executor(None, self.get_next_batch)
                cur_global_steps, gen_batch, batch_dict = next(dataloader_batch_iter)
                if gen_batch == PIPELINE_END_SIGNAL:
                    print("dataloader loop finished.")
                    await self._async_pipeline.push(src_role="dataloader", dst_role="train", data=PIPELINE_END_SIGNAL)
                    break
                
                # 将训练的批次放入训练队列
                await self._async_pipeline.push(src_role="dataloader", dst_role="train", data=(cur_global_steps, batch_dict))
                
                # 将生成的批次放入训练队列
                await self._async_pipeline.push(src_role="train", dst_role="rollout", data=(cur_global_steps, gen_batch))

                next_queue = self._async_pipeline.get_cur_queue(src_role="train", dst_role="rollout")
                print(f"[dataloader] Pushed step:{cur_global_steps}, gs:{self.global_steps} batch to train queue. Next queue size: {next_queue.qsize()}")


    async def rollout_generate(self):
        
        while True:
            t1 = time.time()
            _is_complete = self._async_pipeline.is_complete(src_role="rollout", dst_role="train")
            if _is_complete:
                print(f"[rollout] Pipeline is complete, exiting rollout_generate.")
                break
            
            # check update-param before generate
            cur_model_queue = self._async_pipeline.get_cur_queue(src_role="param_update", dst_role="rollout")
            if cur_model_queue.qsize() > 0:
                cur_model_step = await self._async_pipeline.pull(src_role="param_update", dst_role="rollout")
                print(f"[rollout] Current model step: {cur_model_step}, global steps: {self.global_steps}, generate_global_step:{self.generate_global_step}")
            
            cur_queue = self._async_pipeline.get_cur_queue(src_role="train", dst_role="rollout")
            
            print(f"[rollout] Waiting for training data in the queue... Current queue size: {cur_queue.qsize()}")
            step, gen_batch = await self._async_pipeline.pull(src_role="train", dst_role="rollout")

            next_queue = self._async_pipeline.get_cur_queue(src_role="rollout", dst_role="train")
            
            t2 = time.time()
            # rollout
            print(f"[rollout] Step {step}: Generating sequences, Next queue size: {next_queue.qsize()}")
            # generate_task = asyncio.create_task(self.rollout_wg.async_generate_sequences(gen_batch))
            # 等待生成完成
            # gen_batch_output = await generate_task
            
            # 使用线程池执行同步操作
            gen_batch_output = await asyncio.to_thread(self.rollout_wg.generate_sequences, gen_batch)

            # gen_batch_output = self.rollout_wg.generate_sequences(gen_batch)

            t3 = time.time()
            # # 后处理后的批次
            # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            # # repeat to align with repeated responses in rollout
            # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            # batch = batch.union(gen_batch_output)

            print(f"[rollout] Sending rollout data to train queue, Next queue size: {next_queue.qsize()}")
            await self._async_pipeline.push(src_role="rollout", dst_role="train", data=gen_batch_output)
            self.generate_global_step += 1
            await self._async_pipeline.push(src_role="rollout", dst_role="param_update", data=self.generate_global_step)
            
            t4 = time.time()
            
            print(f"[rollout] Step {step}: Sent rollout data to train queue size: {next_queue.qsize()}. Time taken: {t4 - t1:.2f}s (pull: {t2 - t1:.2f}s, generate: {t3 - t2:.2f}s, push: {t4 - t3:.2f}s)")


    async def rollout_mock(self, mock_data=True):
        for step in range(5):
            print(f"Waiting for training data in the queue...")
            # 获取训练批次
            # train_data = await self.train_to_rollout_queue.get()
            # if type(self.train_to_rollout_queue) is asyncio.Queue:
            #     train_data = self.train_to_rollout_queue.get()
            # else:
            #     # ray.util.queue.Queue.get() 是 同步阻塞调用，而你的 rollout_mock 是一个 async 协程。
            #     # 一旦在 async 协程里执行了同步阻塞函数，事件循环线程就会被卡住
            #     train_data = await self.train_to_rollout_queue.get_async()
            
            train_data = await self._async_pipeline.pull(src_role="train", dst_role="rollout")
            print(f"[Rollout] Step {step + 1}: Received rollout data from train queue")
            
            # rollout
            if mock_data:
                # # rollout_data = self.rollout_wg.generate_sequences(train_data)
                rollout_data = {
                    "responses": torch.randint(0, 2, (8, 10)),  # 模拟生成的响应
                    "prompts": train_data["input_ids"],  # 使用训练数据的输入作为提示
                    "scores": torch.rand(8),  # 模拟分数
                }
                
            else:
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     # Use the executor to run the rollout generation in a separate thread
                #     # future = executor.submit(self.rollout_wg.generate_sequences, gen_batch)
                #     # gen_batch_output = future.result()
                #     coro = self.rollout_wg.async_generate_sequences(self.train_to_rollout_queue, self.rollout_to_train_queue)
                #     future = executor.submit(asyncio.run, coro)
                #     # No need to Wait for the future to complete and get the result
                #     # result = future.result()
                
                rollout_data = self.rollout_wg.generate_sequences(train_data)
            
            
            # 模拟等待一段时间
            # await asyncio.sleep(1)
            
            # 将rollout结果放入队列
            print(f"[Rollout] Step {step + 1}: Sending rollout data to train queue")
            # await self.rollout_to_train_queue.put(rollout_data)
            # self.rollout_to_train_queue.put(rollout_data)
            await self._async_pipeline.push(src_role="rollout", dst_role="train", data=rollout_data)

    async def rollout_logp(self):
        """执行rollout逻辑"""
        # while True:
        #     # 模拟rollout逻辑
        #     print("Performing rollout logp...")
        #     await asyncio.sleep(1)
        
        while True:
            _is_complete = self._async_pipeline.is_complete(src_role="logp", dst_role="train")
            if _is_complete:
                print(f"[Logp] Pipeline is complete, exiting rollout_logp.")
                break
        
            cur_queue = self._async_pipeline.get_cur_queue(src_role="train", dst_role="logp")
            print(f"[Logp] Waiting for training data in the queue... Current queue size: {cur_queue.qsize()}")
            batch = await self._async_pipeline.pull(src_role="train", dst_role="logp")
            
            old_log_prob = self.actor_wg.compute_log_prob(batch)
            
            await self._async_pipeline.push(src_role="logp", dst_role="train", data=old_log_prob)


    async def rollout_ref_logp(self):
        """执行rollout逻辑"""
        # while True:
        #     # 模拟rollout逻辑
        #     print("Performing rollout ref logp...")
        #     await asyncio.sleep(1)
        
        if not self.use_reference_policy:
            return
        
        while True:
            _is_complete = self._async_pipeline.is_complete(src_role="ref_logp", dst_role="train")
            if _is_complete:
                print(f"[Ref Logp] Pipeline is complete, exiting rollout_ref_logp.")
                break
            
            cur_queue = self._async_pipeline.get_cur_queue(src_role="train", dst_role="ref_logp")
            print(f"[Ref Logp] Waiting for training data in the queue... Current queue size: {cur_queue.qsize()}")
            batch = await self._async_pipeline.pull(src_role="train", dst_role="ref_logp")
            
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                
            await self._async_pipeline.push(src_role="ref_logp", dst_role="train", data=ref_log_prob)
            
            # if self.use_reference_policy:
            #     # compute reference log_prob
            #     with _timer("ref", timing_raw):
            #         if not self.ref_in_actor:
            #             ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            #         else:
            #             ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            #         batch = batch.union(ref_log_prob)

    async def rollout_reward_fn(self):
        """执行rollout逻辑"""
        while True:
            _is_complete = self._async_pipeline.is_complete(src_role="reward", dst_role="train")
            if _is_complete:
                print(f"[Reward] Pipeline is complete, exiting rollout_reward_fn.")
                break
        
            t1 = time.time()
            batch = await self._async_pipeline.pull(src_role="train", dst_role="reward")
            t2 = time.time()
            # with _timer("reward", timing_raw):
            # compute reward model score
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                # sync wait for the future
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            t3 = time.time()
            # push to the train queue
            await self._async_pipeline.push(src_role="reward", dst_role="train", data=(reward_tensor, reward_extra_infos_dict))
            t4 = time.time()
            # [Rollout Reward Fn] Step 1: Sent reward to train queue. Time taken: 445.75s (pull: 340.98s, compute: 3.00s, push: 101.77s) 
            print(f"[Reward] Step {self.global_steps}: Sent reward to train queue. Time taken: {t4 - t1:.2f}s (pull: {t2 - t1:.2f}s, compute: {t3 - t2:.2f}s, push: {t4 - t3:.2f}s)")

    async def param_update_loop(self):
        """参数更新循环"""
        while True:
            _is_complete = self._async_pipeline.is_complete(src_role="param_update", dst_role="train")
            if _is_complete:
                print(f"[Param Update] Pipeline is complete, exiting param_update_loop.")
                break
            
            # 等待参数更新：train-step-done -> syncing -> done
            model_step = await self._async_pipeline.pull(src_role="train", dst_role="param_update")
            rollout_step = await self._async_pipeline.pull(src_role="rollout", dst_role="param_update")
            
            print(f"[Param Update] Received model step: {model_step}, rollout step: {rollout_step}")
            if model_step <= rollout_step:
                # 同步权重到所有worker
                await self.sync_weight()
            
            # await self._async_pipeline.push(src_role="param_update", dst_role="train", data=model_step)
            await self._async_pipeline.push(src_role="param_update", dst_role="rollout", data=model_step)
            
            # 更新参数
            print(f"[Param Update] step:{model_step} Parameters updated.")
            
    async def train_loop(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )


        # load checkpoint before doing anything
        self._load_checkpoint()

        # # perform validation before training
        # # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     print(f"===== validation before training =====", flush=True)
        #     val_metrics = self._validate()
        #     assert val_metrics, f"{val_metrics=}"
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # update first param?
        await self._async_pipeline.push(src_role="train", dst_role="param_update", data=self.global_steps)
        
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # 创建生产者的batches，用于进行异步pipeline的数据流构建
        # 这里的生产者是train_loop的主线程
        # 1. 创建一个生产者batches，先进行rollout的输入
        # 2. rollout结束之后进入下一个batch的生成
        # 3. 此时，train用上一个batches进行训练，从而实现overlap
        # warmup一次；
        await self._async_pipeline.push(src_role="train", dst_role="dataloader", data=PIPELINE_START_SINGLE)
        # dataloader_batch_iter = self.get_next_batch()
        # dataloader_batches = next(dataloader_batch_iter)
        # await self._async_pipeline.push(src_role="train", dst_role="rollout", data=dataloader_batches)
        
        
        # for epoch in range(self.config.trainer.total_epochs):
        #     for batch_dict in self.train_dataloader:
        if True:  # async-rl is a loop, so we don't need to loop over epochs and batches
            while True:
                metrics = {}
                timing_raw = {}
                
                # async get batch from the dataloader
                cur_global_steps, batch_dict = await self._async_pipeline.pull(src_role="dataloader", dst_role="train")
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # 需要重复执行预处理部分、
                _gen_batch = self._pre_process_batch(batch)
                
                is_last_step = self.global_steps >= self.total_training_steps
                
                with _timer("step", timing_raw):
                    # async rollout in the background
                    with _timer("gen", timing_raw):
                        # get the batch from the queue
                        gen_batch_output = await self._async_pipeline.pull(src_role="rollout", dst_role="train")

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    
                    with _timer("reward", timing_raw):
                        if self._async_pipeline.is_in_pipeline("reward"):
                            
                            await self._async_pipeline.push("train", "reward", batch)
                            
                            # TODO: lazy load reward_fn
                            # reward_tensor, reward_extra_infos_dict = await self._async_pipeline.pull(src_role="reward", dst_role="train")

                        else:
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if self._async_pipeline.is_in_pipeline("logp"):
                        await self._async_pipeline.push(src_role="train", dst_role="logp", data=batch)
                    if self._async_pipeline.is_in_pipeline("ref_logp"):
                        if self.use_reference_policy:
                            await self._async_pipeline.push(src_role="train", dst_role="ref_logp", data=batch)     

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        if not self._async_pipeline.is_in_pipeline("logp"):
                            # old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            old_log_prob = self.actor_wg.compute_log_prob(batch)
                        else:
                            old_log_prob = await self._async_pipeline.pull(src_role="logp", dst_role="train")
                        
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )
                    
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if self._async_pipeline.is_in_pipeline("ref_logp"):
                                ref_log_prob = await self._async_pipeline.pull(src_role="ref_logp", dst_role="train")
                            else:
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)


                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self._async_pipeline.is_in_pipeline("reward"):
                            reward_tensor, reward_extra_infos_dict = await self._async_pipeline.pull(src_role="reward", dst_role="train")
                        else:
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        batch.batch["token_level_scores"] = reward_tensor

                        # print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # print(f"batch_size:{batch.batch.batch_size[0]}")
                            
                            actor_output = self.actor_wg.update_actor(batch)
                            
                            await self._async_pipeline.push(src_role="train", dst_role="param_update", data=self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    t_post1 = time.time()
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # # validate
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    #     with _timer("testing", timing_raw):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            worker = self.actor_rollout_wg if "actor_rollout" in self.resource_pool_to_cls else self.actor_wg
                            self._save_checkpoint(worker)

                    t_post2 = time.time()
                    print(f"[Train] Step {self.global_steps}: Post-processing took {t_post2 - t_post1:.2f}s")

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": self.epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    async def train_mock(self):
        for step in range(5):
            # 模拟训练逻辑
            train_batch = {
                "input_ids": torch.randint(0, 2, (8, 10)),  # 模拟输入
                "labels": torch.randint(0, 2, (8, 10))  # 模拟标签
            }
            print(f"[Training] Step {step + 1}: Sending batch to rollout queue")

            await self._async_pipeline.push('train', 'rollout', train_batch)
            
            # 模拟等待一段时间
            await asyncio.sleep(1)
            
            # 模拟从rollout队列获取结果
            result = await self._async_pipeline.pull('rollout', 'train')
            print(f"[Training] Step {step + 1}: Received result from rollout queue: {result}")
            
    async def fit_async(self):
        """异步执行rollout和train，简单但增加overlap等逻辑容易混乱，已废弃（切换至状态机的task-loop）"""
        await asyncio.gather(
            # 1. dataloader_loop
            # 2. train_loop get batch from dataloader_loop
            # 3. rollout_generate get batch from train_loop
            # 4. rollout_reward_fn get batch from train_loop
            # 5. rollout_logp get batch from train_loop
            # 6. rollout_ref_logp get batch from train_loop
            # 7. param_update_loop triggered by train_loop/rollout_generate;
            self.dataloader_loop(),
            self.train_loop(),
            self.rollout_generate(),
            self.rollout_reward_fn(),
            self.rollout_logp(),
            self.rollout_ref_logp(),
            self.param_update_loop(),
        )

    def fit(self, use_blocking_mode=False):
        """
        同步入口，启动异步任务
        
        Args:
            use_blocking_mode: 是否使用阻塞模式（默认False）
                              True: 使用同步模式，走nccl做同步(因为受限于线程安全，无法做异步参数同步，所以废弃)
                              False: 使用纯异步模式，走cpu做异步参数同步
        """
        mode_name = "blocking" if use_blocking_mode else "async"
        print(f"Starting async fit with {mode_name} mode...")
        
        from verl.trainer.ppo.pipeline import AsyncTrainingFlow

        enhanced_trainer = AsyncTrainingFlow(
            self, 
            use_async_rl=not use_blocking_mode,
        )
        asyncio.run(enhanced_trainer.run_state_machine_pipeline())
