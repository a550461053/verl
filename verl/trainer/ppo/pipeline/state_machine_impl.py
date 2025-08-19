"""
State machine implementation with state-machine for n-step off-policy Async-RL training.

This implementation ensures strict dependency relationships between modules by using blocking
operations queue. This prevents errors and ensures correct one-step or n-step
off-policy training flow.

Key Features:
- Strict dependencies: Ensures proper order of operations
- State machine design: Clear separation of concerns, each task has its own loop state machine to handle its logic
- N-step off-policy: Maintains correct training flow for RL algorithms
"""

import asyncio
import time
import torch
import uuid
from enum import Enum, auto
from typing import Optional, Any, Dict, List
from abc import ABC, abstractmethod
from tqdm import tqdm
from pprint import pprint

from .pipeline_utils import AsyncPipeline, enhanced_print, PIPELINE_END_SIGNAL, PIPELINE_START_SINGLE, TransferMode
from .state_machine import BaseRoleStateMachine, RoleState, RoleEvent
from .utils import resource_lock, global_timing_collector, timing_decorator

import ray
from ray.util.queue import Queue
import numpy as np
from copy import deepcopy

from verl.utils.metric import (
    reduce_metrics,
)
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


class DataloaderStateMachine(BaseRoleStateMachine):
    """Enhanced dataloader state machine, fully aligned with ray_async_pipeline_trainer.py switching conditions"""
    # input:
    #   rollout -> dataloader (just start signal once)
    # output:
    #   dataloader -> rollout
    #   dataloader -> generate
    #   dataloader -> train
    #   dataloader -> param_update
    #   dataloader -> logp
    def __init__(self, pipeline, trainer):
        super().__init__("dataloader", pipeline)
        self.trainer = trainer
        self.batch_iter = None
        self.pipeline_start = None  # Store startup signal
        
        # Configure dataloader prefetch - increase default value to accommodate larger queues
        self.prefetch_steps = trainer.config.trainer.get("dataloader_prefetch_steps", 10)  # Increased from 4 to 10
        self.max_pending_size = self.prefetch_steps
        enhanced_print("dataloader", None, f"Configured dataloader prefetch: {self.prefetch_steps} steps")
    
    async def get_input_data(self) -> Optional[Any]:
        """Block getting batch data, ensure dependencies"""
        # Check if startup signal exists
        if not self.pipeline_start:
            enhanced_print("dataloader", None, "Waiting for start signal from rollout")
            # Block waiting for startup signal
            signal = await self.pipeline.pull("rollout", "dataloader")
            enhanced_print("dataloader", None, f"Received signal from rollout: {signal}")
            if signal == PIPELINE_START_SINGLE:
                self.pipeline_start = signal
                enhanced_print("dataloader", None, "Pipeline started, Initializing batch iterator")
                self.batch_iter = self.trainer.get_next_batch()
        
        # Check queue size, block waiting for queue space
        queue_size = self.pipeline.get_queue_size("dataloader", "generate")
        if queue_size >= self.max_pending_size:
            # Queue full, block waiting
            enhanced_print("dataloader", None, f"Queue full, waiting... size: {queue_size}")
            # Use blocking wait instead of sleep
            while self.pipeline.get_queue_size("dataloader", "generate") >= self.max_pending_size:
                await asyncio.sleep(0.1)  # Brief check interval
            enhanced_print("dataloader", None, f"Queue has space, continuing... size: {self.pipeline.get_queue_size('dataloader', 'generate')}")
        
        enhanced_print("dataloader", None, "Returning START")
        return "START"
    
    @timing_decorator("dataloader")
    async def process_data(self, data: Any) -> Any:
        """Process data loading logic - block execution to ensure correctness"""
        if data == "START":
            # Block getting next batch
            batch_result = next(self.batch_iter)
            
            cur_global_steps, gen_batch, batch_dict = batch_result
            if gen_batch == PIPELINE_END_SIGNAL:
                enhanced_print("dataloader", None, "dataloader loop finished.")
                return "END"
            
            # Validate batch_dict data
            if not batch_dict or not isinstance(batch_dict, dict):
                enhanced_print("dataloader", None, f"Invalid batch_dict from trainer: {batch_dict}")
                return None
            
            enhanced_print("dataloader", None, f"Returning batch for step {cur_global_steps}")
            return (cur_global_steps, gen_batch, batch_dict)
        else:
            enhanced_print("dataloader", None, f"Unexpected data received: {data}, returning None")
            return None
    
    async def send_output_data(self, data: Any) -> bool:
        """Send data to trainer - ensure no data loss"""
        if data == "END":
            # Send END signal
            await self.pipeline.push("dataloader", "rollout", PIPELINE_END_SIGNAL)
            await self.pipeline.push("dataloader", "generate", PIPELINE_END_SIGNAL)
            enhanced_print("dataloader", None, "Sent END signals to rollout and generate")
            return True
        elif isinstance(data, tuple):
            cur_global_steps, gen_batch, batch_dict = data
            # Push data, ensure no loss
            await self.pipeline.push("dataloader", "rollout", (cur_global_steps, batch_dict))
            await self.pipeline.push("dataloader", "generate", (cur_global_steps, gen_batch))
            enhanced_print("dataloader", None, f"Sent batch for step {cur_global_steps} to rollout and generate")
            return True
        else:
            enhanced_print("dataloader", None, f"Unexpected data: {type(data)}")
        return False


class RolloutStateMachine(BaseRoleStateMachine):
    # Receive raw data from dataloader
    # input: 
    #   dataloader -> rollout
    #   generate -> rollout
    # output:
    #   rollout -> ref_logp
    #   rollout -> logp
    #   rollout -> reward
    #   rollout -> train
    def __init__(self, pipeline, trainer):
        super().__init__("rollout", pipeline)
        self.trainer = trainer
        
        # Configure parameters
        rollout_wg = trainer.rollout_wg
        self._total_engines = rollout_wg.world_size
        tp_size = trainer.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self._tp_rank_0_engines = self._total_engines // tp_size
        
        # Check if resources are shared, TODO: support separated train vs logp/ref_logp
        self._share_resources = trainer.config.trainer.get("share_resource_between_train_logp_ref_logp", True)
    
    async def get_input_data(self) -> Optional[Any]:
        """Block getting dataloader; wait for generate data to arrive"""
        import asyncio

        data = await self.pipeline.pull("dataloader", "rollout")

        if data == PIPELINE_END_SIGNAL:
            return "END"
        
        # dataloader data arrives, cache and check if generate for corresponding step is already cached
        cur_global_steps, train_batch = data

        gen_step, gen_batch_output = await self.pipeline.pull("generate", "rollout")
        enhanced_print("rollout", None, f"Received data of gen_step: {gen_step}, dataloader step: {cur_global_steps}")

        return (cur_global_steps, train_batch, gen_batch_output)
    
    @timing_decorator("rollout")
    async def process_data(self, data: Any) -> Any:
        """Process training logic - block execution to ensure correctness"""
        if data is None:
            enhanced_print("rollout", None, "Received None data, waiting...")
            return None
        if data == "END":
            return {"step": None, "batch_dict": None, "batch": None, "pipeline_signal": PIPELINE_END_SIGNAL}
        
        # Handle two cases: only dataloader data, or dataloader+generate data
        assert isinstance(data, (int, tuple, list)) and len(data) == 3, f"Invalid data format: {data}"
        # Check if it's (dataloader_data, generate_data) format
        cur_global_steps, train_batch, gen_batch_output = data
        batch_dict = train_batch
        
        metrics = {}
        timing_raw = {}
        
        # Block process batch data
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        # Need to repeat preprocessing part
        _gen_batch = self.trainer._pre_process_batch(batch)
        
        with _timer("step", timing_raw):
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.trainer.config.actor_rollout_ref.rollout.n, interleave=True)
            
            # If there's generate data, merge it
            if gen_batch_output is not None:
                batch = batch.union(gen_batch_output)

            batch.batch["response_mask"] = compute_response_mask(batch)
            # Balance the number of valid tokens across DP ranks.
            # NOTE: This usually changes the order of data in the `batch`,
            # which won't affect the advantage calculation (since it's based on uid),
            # but might affect the loss calculation (due to the change of mini-batching).
            # TODO: Decouple the DP balancing and mini-batching.
            if self.trainer.config.trainer.balance_batch:
                self.trainer._balance_batch(batch, metrics=metrics)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            
            # Hand over to train loop
        
        return {
            "step": cur_global_steps,
            "batch": batch,
            "batch_dict": batch_dict,
            "pipeline_signal": PIPELINE_START_SINGLE,
        }

    
    async def send_output_data(self, data: Any) -> bool:
        """Send training results - ensure no data loss"""
        pipeline_signal = data["pipeline_signal"]
        if pipeline_signal == PIPELINE_END_SIGNAL:
            batch = PIPELINE_END_SIGNAL
        else:
            batch = data["batch"]
        
        # Push to all downstream, ensure no data loss
        push_tasks = []
        
        if self.pipeline.is_in_pipeline("logp"):
            push_tasks.append(self.pipeline.push("rollout", "logp", (data["step"], batch)))
        
        if self.pipeline.is_in_pipeline("ref_logp"):
            push_tasks.append(self.pipeline.push("rollout", "ref_logp", (data["step"], batch)))
        
        if self.pipeline.is_in_pipeline("reward"):
            push_tasks.append(self.pipeline.push("rollout", "reward", (data["step"], batch)))
        
        # Push to train
        push_tasks.append(self.pipeline.push("rollout", "train", data))
        
        # Execute all pushes concurrently
        if push_tasks:
            await asyncio.gather(*push_tasks)
            enhanced_print("rollout", None, f"Successfully pushed to {len(push_tasks)} downstream")
        
        enhanced_print("rollout", None, f"Sent step {data['step']} to downstream")
        return True


class TrainStateMachine(BaseRoleStateMachine):
    """Enhanced trainer state machine, fully aligned with ray_async_pipeline_trainer.py switching conditions"""
    # input: 
    #   rollout -> train
    #   logp -> train
    #   ref_logp -> train
    #   reward -> train
    # output:
    #   train -> param_update
    def __init__(self, pipeline, trainer):
        super().__init__("train", pipeline)
        self.trainer = trainer
        # Lazy initialization, avoid calling asyncio.run in constructor
        self._init_completed = False
            
    async def _init_before_train(self):

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.trainer.config.trainer.project_name,
            experiment_name=self.trainer.config.trainer.experiment_name,
            default_backend=self.trainer.config.trainer.logger,
            config=OmegaConf.to_container(self.trainer.config, resolve=True),
        )

        # load checkpoint before doing anything
        self.trainer._load_checkpoint()

        # TODO: support async-rl validation
        # # perform validation before training
        # # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.trainer.config.trainer.get("val_before_train", True):
        #     print(f"===== validation before training =====", flush=True)
        #     val_metrics = self._validate()
        #     assert val_metrics, f"{val_metrics=}"
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     self.logger.log(data=val_metrics, step=self.global_steps)
        #     if self.trainer.config.trainer.get("val_only", False):
        #         return

        # add tqdm
        self.progress_bar = tqdm(total=self.trainer.total_training_steps, initial=self.trainer.global_steps, desc="Training Progress")

        # we start from step 1
        self.trainer.global_steps += 1
        self.last_val_metrics = None

    async def get_input_data(self) -> Optional[Any]:
        """Block getting batch data, ensure dependencies"""
        
        # Ensure initialization is complete
        if not self._init_completed:
            await self._init_before_train()
            self._init_completed = True
        
        # Block waiting for rollout data
        data = await self.pipeline.pull("rollout", "train")
        
        if data is None:
            return None
        if data["pipeline_signal"] == PIPELINE_END_SIGNAL:
            return "END"
        
        # Block waiting for all dependency data
        logp_result = await self.pipeline.pull("logp", "train")
        ref_logp_result = await self.pipeline.pull("ref_logp", "train")
        reward_result = await self.pipeline.pull(src_role="reward", dst_role="train")
        
        # Check if all data is received
        if logp_result is None or ref_logp_result is None or reward_result is None:
            return None
            
        logp_step, old_log_prob = logp_result
        ref_logp_step, ref_log_prob = ref_logp_result
        reward_step, reward_tensor, reward_extra_infos_dict = reward_result
        
        # Verify step consistency
        assert logp_step == ref_logp_step == reward_step, f"Step mismatch: logp_step={logp_step}, ref_logp_step={ref_logp_step}, reward_step={reward_step}"
        
        enhanced_print("train", None, f"Successfully assembled data for step {logp_step}")
        return (data, old_log_prob, ref_log_prob, reward_tensor, reward_extra_infos_dict)
    
    @timing_decorator("train")
    async def process_data(self, data: Any) -> Any:
        """Process training logic, fully aligned with ray_async_pipeline_trainer.py training flow"""
        if data == "END":
            return "END"
        
        data, old_log_prob, ref_log_prob, reward_tensor, reward_extra_infos_dict = data
        batch = data["batch"]
        cur_global_steps = data["step"]
        enhanced_print("train", None, f"Processing step {cur_global_steps}, train_step: {self.trainer.global_steps}")
        
        metrics = {}
        timing_raw = {}
        
        is_last_step = self.trainer.global_steps >= self.trainer.total_training_steps
        
        with _timer("step", timing_raw):
            # Acquire resource lock (for training phase)
            await resource_lock.acquire("train", self.trainer.global_steps)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.trainer.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                
                # Merge old_log_prob to batch
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
            
            if self.trainer.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    batch = batch.union(ref_log_prob)

            # compute values; TODO support async-rl
            if self.trainer.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            with _timer("adv", timing_raw):
                batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                # compute rewards. apply_kl_penalty if available
                if self.trainer.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.trainer.config.algorithm.kl_penalty)
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.trainer.config.algorithm.adv_estimator,
                    gamma=self.trainer.config.algorithm.gamma,
                    lam=self.trainer.config.algorithm.lam,
                    num_repeat=self.trainer.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    multi_turn=self.trainer.config.actor_rollout_ref.rollout.multi_turn.enable,
                    use_pf_ppo=self.trainer.config.algorithm.use_pf_ppo,
                    pf_ppo_reweight_method=self.trainer.config.algorithm.pf_ppo.reweight_method,
                    pf_ppo_weight_pow=self.trainer.config.algorithm.pf_ppo.weight_pow,
                )

            # update critic
            if self.trainer.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.trainer.config.trainer.critic_warmup <= self.trainer.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.trainer.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.trainer.actor_wg.update_actor(batch)

                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.trainer.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with _timer("dump_rollout_generations", timing_raw):
                    print(batch.batch.keys())
                    inputs = self.trainer.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.trainer.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    self.trainer._dump_generations(
                        inputs=inputs,
                        outputs=outputs,
                        scores=scores,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                        dump_path=rollout_data_dir,
                    )

            if self.trainer.config.trainer.save_freq > 0 and (is_last_step or self.trainer.global_steps % self.trainer.config.trainer.save_freq == 0):
                with _timer("save_checkpoint", timing_raw):
                    worker = self.trainer.actor_rollout_wg if "actor_rollout" in self.trainer.resource_pool_to_cls else self.trainer.actor_wg
                    self.trainer._save_checkpoint(worker)

        # Release resource lock
        await resource_lock.release("train", self.trainer.global_steps)

        # training metrics
        metrics.update(
            {
                "training/global_step": self.trainer.global_steps,
                "training/epoch": self.trainer.epoch,
            }
        )
        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.trainer.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.trainer.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        # TODO: make a canonical logger that supports various backend
        self.logger.log(data=metrics, step=self.trainer.global_steps)

        self.progress_bar.update(1)
        self.trainer.global_steps += 1
        if is_last_step:
            pprint(f"Final validation metrics: {self.last_val_metrics}")
            self.progress_bar.close()
            return "END"
        
        # Return the current completed training step, representing model_steps
        return self.trainer.global_steps - 1
    
    async def send_output_data(self, data: Any) -> bool:
        """Send training results - block to ensure data transfer"""
        if data == "END":
            data = PIPELINE_END_SIGNAL

        global_steps = data
        # Block push to param_update, ensure parameter update
        await self.pipeline.push(src_role="train", dst_role="param_update", data=global_steps)
        enhanced_print("train", None, f"Sent training completion signal for step {global_steps} to param_update")

        # If train/logp/ref_logp share resources, block push to logp/ref_logp to ensure no resource contention
        if self.trainer.config.trainer.get("share_resource_between_train_logp_ref_logp", True):
            await self.pipeline.push(src_role="train", dst_role="logp", data=global_steps)
            await self.pipeline.push(src_role="train", dst_role="ref_logp", data=global_steps)
            enhanced_print("train", None, f"Sent training completion signal for step {global_steps} to logp/ref_logp")
        
        return True


class RewardStateMachine(BaseRoleStateMachine):
    """Enhanced reward state machine, using blocking mode to ensure dependencies"""
    # input: 
    #   rollout -> reward
    # output:
    #   reward -> train
    
    def __init__(self, pipeline, trainer):
        super().__init__("reward", pipeline)
        self.trainer = trainer
        
    async def get_input_data(self) -> Optional[Any]:
        """Block getting reward calculation data, ensure dependencies"""
        try:
            # Block waiting for data, no timeout set
            reward_data = await self.pipeline.pull("rollout", "reward")
            if reward_data is None:
                enhanced_print("reward", None, "Received None from rollout, waiting...")
                return None
            return reward_data
        except Exception as e:
            enhanced_print("reward", None, f"Error in get_input_data: {e}")
            return None
    
    @timing_decorator("reward")
    async def process_data(self, data: Any) -> Any:
        """Process reward calculation logic - block execution to ensure correctness"""
        if data is None:
            enhanced_print("reward", None, "Received None data, waiting...")
            return None
        
        step, batch = data
        if batch == PIPELINE_END_SIGNAL:
            return "END"
        enhanced_print("reward", None, f"Computing reward for step {step}")
        
        # Initialize variables
        reward_tensor = None
        reward_extra_infos_dict = {}
        
        # Block execute reward calculation
        if self.trainer.use_rm:
            reward_tensor = self.trainer.rm_wg.compute_rm_score(batch)

        if self.trainer.config.reward_model.launch_reward_fn_async:
            future_reward = compute_reward_async.remote(batch, self.trainer.config, self.trainer.tokenizer)
            # Need to wait for future_reward to complete, but now use sync version first
            enhanced_print("reward", None, "Using async reward computation - converting to sync")
            # reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.trainer.reward_fn)
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.trainer.reward_fn)

        return (step, reward_tensor, reward_extra_infos_dict)
    
    async def send_output_data(self, data: Any) -> bool:
        """Send reward results - block to ensure data transfer"""
        if data is None:
            return False
        if data == "END":
            # no need to send to train queue
            return True
        step, reward_tensor, reward_extra_infos_dict = data
        # Block send results, ensure train can receive data
        await self.pipeline.push("reward", "train", (step, reward_tensor, reward_extra_infos_dict))
        enhanced_print("reward", None, f"Sent reward result for step {step}")
        return True


class LogPStateMachine(BaseRoleStateMachine):
    """Enhanced LogP state machine, using blocking mode to ensure dependencies"""
    # input: 
    #   rollout -> logp
    # output:
    #   logp -> train
    
    def __init__(self, pipeline, trainer):
        super().__init__("logp", pipeline)
        self.trainer = trainer
        
    async def get_input_data(self) -> Optional[Any]:
        """Block getting LogP calculation data, ensure dependencies"""
        try:
            # Block waiting for data, no timeout set
            batch = await self.pipeline.pull("rollout", "logp")

            if batch is None:
                enhanced_print("logp", None, "Received None from rollout, waiting...")
                return None
            
            if batch == PIPELINE_END_SIGNAL:
                return "END"
            
            # If logp/ref_logp and train share resources, wait for train's previous step to complete
            if self.trainer.config.trainer.get("share_resource_between_train_logp_ref_logp", True):
                enhanced_print("logp", None, f"Waiting for train to complete")
                await self.pipeline.pull("train", "logp")
                
                enhanced_print("logp", None, f"Train step completed, continuing with logp")
            return batch
        except Exception as ex:
            enhanced_print("logp", None, f"Error in get_input_data: {ex}")
            return None
    
    @timing_decorator("logp")
    async def process_data(self, data: Any) -> Any:
        """Process LogP calculation logic - block execution to ensure correctness"""
        if data is None:
            enhanced_print("logp", None, "Received None data, waiting...")
            return None
        
        if data == "END":
            return "END"
        step, batch = data
        if batch == PIPELINE_END_SIGNAL:
            return "END"
        enhanced_print("logp", None, f"Computing logp for step {step}")
        
        # Acquire resource lock, pass step parameter
        await resource_lock.acquire("logp", step)
        
        # Block execute LogP calculation (direct call, not using asyncio.to_thread)
        enhanced_print("logp", None, f"Starting LogP computation for step {step}")
        old_log_prob = self.trainer.actor_wg.compute_log_prob(batch)
        enhanced_print("logp", None, f"LogP computation finished for step {step}")
        
        # Release resource lock
        await resource_lock.release("logp")
        
        return (step, old_log_prob)
    
    async def send_output_data(self, data: Any) -> bool:
        """Send LogP results - block to ensure data transfer"""
        if data is None:
            return False
        if data == "END":
            # no need to send to train queue
            return True
        
        step, logp_result = data
        # Block send results, ensure train can receive data
        await self.pipeline.push("logp", "train", (step, logp_result))
        enhanced_print("logp", None, f"Sent logp result for step {step}")

        if self.trainer.config.trainer.get("share_resource_between_train_logp_ref_logp", True):
            await self.pipeline.push("logp", "ref_logp", step)
        return True


class RefLogPStateMachine(BaseRoleStateMachine):
    """Enhanced reference LogP state machine, using blocking mode to ensure dependencies"""
    # input: 
    #   rollout -> ref_logp
    # output:
    #   ref_logp -> train
    def __init__(self, pipeline, trainer):
        super().__init__("ref_logp", pipeline)
        self.trainer = trainer
        
    async def get_input_data(self) -> Optional[Any]:
        """Block getting reference LogP calculation data, ensure dependencies"""
        try:
            # Block waiting for data, no timeout set
            batch = await self.pipeline.pull("rollout", "ref_logp")
            if batch == PIPELINE_END_SIGNAL:
                return "END"

            if batch is None:
                enhanced_print("ref_logp", None, "Received None from rollout, waiting...")
                return None
            
            if self.trainer.config.trainer.get("share_resource_between_train_logp_ref_logp", True):
                enhanced_print("ref_logp", None, f"Waiting for train to complete")
                await self.pipeline.pull("train", "ref_logp")
                await self.pipeline.pull("logp", "ref_logp")
                enhanced_print("ref_logp", None, f"Train step completed, continuing with ref_logp")
            return batch
        except Exception as ex:
            enhanced_print("ref_logp", None, f"Error in get_input_data: {ex}")
            return None
    
    @timing_decorator("ref_logp")
    async def process_data(self, data: Any) -> Any:
        """Process reference LogP calculation logic - block execution to ensure correctness"""
        if data is None:
            enhanced_print("ref_logp", None, "Received None data, waiting...")
            return None
        
        if data == "END":
            return "END"
        
        step, batch = data
        if batch == PIPELINE_END_SIGNAL:
            return "END"
        enhanced_print("ref_logp", None, f"Computing ref_logp for step {step}")

        # Acquire resource lock, pass step parameter
        await resource_lock.acquire("ref_logp", step)
        
        # Block execute reference LogP calculation (direct call, not using asyncio.to_thread)
        enhanced_print("ref_logp", None, f"Starting Ref LogP computation for step {step}")
        if not self.trainer.ref_in_actor:
            ref_log_prob = self.trainer.ref_policy_wg.compute_ref_log_prob(batch)
        else:
            ref_log_prob = self.trainer.actor_wg.compute_ref_log_prob(batch)
        enhanced_print("ref_logp", None, f"Ref LogP computation finished for step {step}")
        
        # Release resource lock
        await resource_lock.release("ref_logp")
        
        return (step, ref_log_prob)
    
    async def send_output_data(self, data: Any) -> bool:
        """Send reference LogP results - block to ensure data transfer"""
        if data is None:
            return False
        if data == "END":
            # no need to send to train queue
            return True
        
        step, ref_logp_result = data
        # Block send results, ensure train can receive data
        await self.pipeline.push("ref_logp", "train", (step, ref_logp_result))
        enhanced_print("ref_logp", None, f"Sent ref_logp result for step {step}")
        return True


class ParamUpdateStateMachine(BaseRoleStateMachine):
    """asyncRLparameterupdatestate machine - 使用param_update.py中的方法进行asyncsynchronization"""
    
    def __init__(self, pipeline, trainer):
        super().__init__("param_update", pipeline)
        self._debug = False  # sync update by all params
        self.trainer = trainer
        self.stats = {
            "updates": 0,
            "async_updates": 0,
            "sync_updates": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0
        }
        
        # 检查trainer是否有param_update_manager
        rollout_wg = self.trainer.rollout_wg
        actor_wg = self.trainer.actor_wg
        self.has_param_update_manager = hasattr(actor_wg, 'async_param_update')
        
        if self.has_param_update_manager:
            enhanced_print("AsyncRLParamUpdate", None, "Using param_update_manager for async parameter synchronization")
        else:
            enhanced_print("AsyncRLParamUpdate", None, "param_update_manager not available, falling back to sync update")
    
    async def get_input_data(self) -> Optional[Any]:
        """gettingparameterupdate请求"""
        try:
            data = await self.pipeline.pull("train", "param_update")
            if data == PIPELINE_END_SIGNAL:
                return "END"
            elif data is None:
                enhanced_print("param_update", None, "Received None from train, waiting...")
                return None
            
            enhanced_print("param_update", None, f"Received param update request for step {data}")
            return data
            
        except Exception as e:
            enhanced_print("param_update", None, f"Error in get_input_data: {e}")
            return None
    
    @timing_decorator("param_update")
    async def process_data(self, data: Any) -> Any:
        """Processparameterupdate请求 - 后台线程execute"""
        if data == "END":
            return "END"
        elif data is None:
            enhanced_print("param_update", None, "Received None data, waiting...")
            return None
        
        global_steps = data

        enhanced_print("param_update", None, f"Starting param update for step {global_steps}")
        
        # 记录开始时间
        start_time = time.time()
        # 后续step在后台线程中execute，不阻塞主流程
        if self.has_param_update_manager:
            # 启动后台asyncparameterupdate任务
            param_update_task = asyncio.create_task(
                self._perform_async_param_update_background(global_steps)
            )
            self.stats["async_updates"] += 1
            enhanced_print("param_update", None, f"Async param update task created for step {global_steps}")
        else:
            # 启动后台synchronizationparameterupdate任务
            param_update_task = asyncio.create_task(
                asyncio.to_thread(self._perform_sync_param_update_background, global_steps)
            )
            self.stats["sync_updates"] += 1
            enhanced_print("param_update", None, f"Sync param update task created for step {global_steps}")
        
        # 立即return，不waitingparameterupdatecompleted
        task_creation_time = time.time() - start_time
        
        enhanced_print("param_update", None, 
                        f"Param update task created for step {global_steps} in {task_creation_time:.3f}s (background execution)")
        
        # return任务对象，让send_output_datawaitingcompleted
        return (global_steps, param_update_task)

    async def _perform_async_param_update_background(self, global_steps: int) -> bool:
        """后台asyncparameterupdate - 后续step使用"""
        enhanced_print("param_update", None, f"Background async param update started for step {global_steps}")
        
        # getting锁（在后台线程中）
        await resource_lock.acquire("param_update", global_steps)

        start_time = time.time()
        
        # 使用param_update_manager的async_param_update方法
        self.trainer.actor_wg.async_param_update()
        self.trainer.rollout_wg.async_param_update()
        
        # waitingsendcompleted
        self.trainer.actor_wg.wait_for_send_complete()
        
        update_time = time.time() - start_time
        
        # update统计
        self.stats["updates"] += 1
        self.stats["total_time"] += update_time
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["updates"]
        self.stats["min_time"] = min(self.stats["min_time"], update_time)
        self.stats["max_time"] = max(self.stats["max_time"], update_time)
        
        enhanced_print("param_update", None, 
                        f"Background async param update completed for step {global_steps} in {update_time:.3f}s")
        
        # 释放锁
        await resource_lock.release("param_update", global_steps)

        return True
            

    async def _perform_sync_param_update_background(self, global_steps: int) -> bool:
        enhanced_print("param_update", None, f"Background sync param update started for step {global_steps}")
        
        await resource_lock.acquire("param_update", global_steps)
        
        start_time = time.time()
        
        def sync_param_update():
            enhanced_print("param_update", None, f"Syncing actor parameters for step {global_steps}...")
            actor_result = self.trainer.actor_wg.sync_per_tensor_generator()
            
            enhanced_print("param_update", None, f"Syncing rollout parameters for step {global_steps}...")
            rollout_result = self.trainer.rollout_wg.sync_per_tensor_generator()
            
            if hasattr(rollout_result, '__class__') and 'ObjectRef' in str(type(rollout_result)):
                ray.get(rollout_result)
            
            enhanced_print("param_update", None, f"Parameter sync completed for step {global_steps}")
            return True
        
        success = await asyncio.to_thread(sync_param_update)
        
        update_time = time.time() - start_time
        
        self.stats["updates"] += 1
        self.stats["total_time"] += update_time
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["updates"]
        self.stats["min_time"] = min(self.stats["min_time"], update_time)
        self.stats["max_time"] = max(self.stats["max_time"], update_time)
        
        enhanced_print("param_update", None, 
                        f"Background sync param update completed for step {global_steps} in {update_time:.3f}s")
        
        await resource_lock.release("param_update", global_steps)
        
        return success
        

    async def send_output_data(self, data: Any) -> bool:
        """sendupdatecompleted信号 - waitingasync任务completed后push"""
        if data == "END":
            enhanced_print("param_update", None, "Async param update completed, END signal processed")
            await self.pipeline.push("param_update", "generate", PIPELINE_END_SIGNAL)
            return True
        elif data is None:
            return False
        
        # 检查是否是任务对象
        if isinstance(data, tuple) and len(data) == 2:
            global_steps, param_update_task = data
            enhanced_print("param_update", None, f"Waiting for background param update task to complete for step {global_steps}")
            
            # waiting后台任务completed
            success = await param_update_task
            if success:
                enhanced_print("param_update", None, f"Background param update completed for step {global_steps}")
            else:
                enhanced_print("param_update", None, f"Background param update failed for step {global_steps}")
        else:
            # 第一个step的情况，data就是global_steps
            global_steps = data
        
        # unified在这里Processpush操作
        enhanced_print("param_update", None, f"Sending completion signal to generate for step {global_steps}")
        await self.pipeline.push("param_update", "generate", global_steps)
        enhanced_print("param_update", None, f"Sent completion signal to generate for step {global_steps}")
        return True
    
    def get_status_info(self) -> Dict[str, Any]:
        async_rl_stats = {}
        if self.has_param_update_manager and hasattr(self.trainer, 'param_update_manager'):
            async_rl_stats = self.trainer.param_update_manager.get_async_rl_stats()
        
        return {
            "stats": self.stats.copy(),
            "type": "async_param_update",
            "has_param_update_manager": self.has_param_update_manager,
            "async_rl_stats": async_rl_stats,
            "description": "Async param update with param_update_manager"
        }


class GenerateStateMachine(BaseRoleStateMachine):
    """asyncRLgenerationstate machine - 支持可interruptgeneration"""
    
    def __init__(self, pipeline, trainer):
        super().__init__("generate", pipeline)
        self.trainer = trainer
        self.first_generation = True  # 标记是否为第一次generation
        
        # 配置generate与param_update的距离
        self.generate_ahead_steps = trainer.config.trainer.get("generate_ahead_steps", 3)  # 增加默认值
        self.last_param_update_step = 0  # 记录最后一个param_update的step

        enhanced_print("generate", None, f"Configured generate ahead: {self.generate_ahead_steps} steps")

    @timing_decorator("generate")
    async def process_data(self, data: Any) -> Any:
        """Processgenerationlogic - 后台线程execute"""
        if data == "END":
            return "END"
        elif data is None:
            enhanced_print("generate", None, "Received None data, waiting...")
            return None
        
        step, gen_batch = data
        enhanced_print("generate", None, f"Starting generation task for step {step}")
        
        # 在后台线程中executegeneration，不阻塞主流程
        generation_task = asyncio.create_task(
            asyncio.to_thread(self._generate_sync, gen_batch, step)
        )
        
        # updategenerate_global_step
        self.trainer.generate_global_step += 1
        
        enhanced_print("generate", None, f"Generation task created for step {step}")
        
        # return任务对象，让send_output_datawaitingcompleted
        return (step, generation_task)

    async def send_output_data(self, data: Any) -> bool:
        """sendgenerationresults - waitingasync任务completed后push"""
        if data == "END":
            enhanced_print("generate", None, "Sending END signal to rollout")
            await self.pipeline.push("generate", "rollout", PIPELINE_END_SIGNAL)
            return True
        elif data is None:
            return False
        
        # 检查是否是任务对象
        if isinstance(data, tuple) and len(data) == 2:
            step, generation_task = data
            enhanced_print("generate", None, f"Waiting for background generation task to complete for step {step}")
            
            # waiting后台任务completed
            gen_batch_output = await generation_task
            if gen_batch_output is None:
                raise Exception(f"Generation failed for step {step}")
            
            enhanced_print("generate", None, f"Background generation completed for step {step}")

        else:
            # Process其他情况
            step, gen_batch_output = data
        
        # unified在这里Processpush操作
        enhanced_print("generate", None, f"Sending generation result to rollout for step {step}")
        await self.pipeline.push("generate", "rollout", (step, gen_batch_output))
        enhanced_print("generate", None, f"Generation result sent to rollout for step {step}")
        
        return True
    
    async def get_input_data(self) -> Optional[Any]:
        """gettinggenerationdata - 只在距离过大时才waitingparam_update"""        
        # 第一次generation时需要waitingparam_updatecompleted，确保有可用的权重
        if self.first_generation:
            enhanced_print("generate", None, "First generation, waiting for initial param_update to complete...")
            # 第一个step时，必须waitingparam_updatecompleted
            param_update_signal = await self.pipeline.pull("param_update", "generate")
            
            if param_update_signal == PIPELINE_END_SIGNAL:
                enhanced_print("generate", None, "Received END signal from param_update")
                return "END"
            elif param_update_signal is None:
                return None
            
            # update最后一个param_update的step
            self.last_param_update_step = param_update_signal
            enhanced_print("generate", None, f"First generation: received param_update completion signal for step {param_update_signal}")
            self.first_generation = False
        
        # waitingdataloader的data
        data = await self.pipeline.pull("dataloader", "generate")
        
        if data == PIPELINE_END_SIGNAL:
            enhanced_print("generate", None, "Received END signal from dataloader")
            return "END"
        elif data is None:
            return None
        
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            enhanced_print("generate", None, f"Invalid data format: {data}")
            return None
            
        step, gen_batch = data
        
        # 检查generate与param_update的距离，如果太远则Block waitingparam_update来updatelast_param_update_step
        while step >= self.last_param_update_step + self.generate_ahead_steps:
            enhanced_print("generate", None, f"Step {step} is too far ahead of param_update {self.last_param_update_step}, waiting for next param_update...")
            # waiting下一个param_updatecompleted来updatelast_param_update_step
            param_update_signal = await self.pipeline.pull("param_update", "generate")
            
            if param_update_signal == PIPELINE_END_SIGNAL:
                enhanced_print("generate", None, "Received END signal from param_update while waiting")
                return "END"
            elif param_update_signal is None:
                return None
            
            # update最后一个param_update的step
            self.last_param_update_step = param_update_signal
            enhanced_print("generate", None, f"Updated param_update step to {param_update_signal}")
        
        enhanced_print("generate", None, f"Got generation task for step {step}")
        return (step, gen_batch)
    
    def _generate_sync(self, gen_batch, step: int):
        """synchronizationgeneration函数 - 在后台线程中execute"""
        try:
            enhanced_print("generate", None, f"Background generation started for step {step}")
            
            start_time = time.time()
            
            # executegeneration
            wg = self.trainer.rollout_wg
            gen_batch_output = wg.generate_sequences_sperated(gen_batch)
            
            generation_time = time.time() - start_time
            
            enhanced_print("generate", None, f"Background generation completed for step {step} in {generation_time:.3f}s")
            
            return gen_batch_output
            
        except Exception as e:
            enhanced_print("generate", None, f"Error in background generation for step {step}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_status_info(self) -> Dict[str, Any]:
        """getting详细状态信息"""
        async_rl_stats = {}
        return {
            "type": "async_rl_generate",
            "has_async_rl_support": True,
            "async_rl_stats": async_rl_stats,
            "description": "Async RL generate with interruptible generation"
        }


def create_role_state_machine(role_name: str, pipeline, trainer, use_async_rl: bool = False) -> BaseRoleStateMachine:
    """
    创建角色state machine工厂函数
    
    Args:
        role_name: 角色名称
        pipeline: pipeline实例
        trainer: trainer实例  
        use_async_rl: Whether to use async RL optimization (default False)
    """
    state_machines = {
        "dataloader": DataloaderStateMachine,
        "rollout": RolloutStateMachine,
        "reward": RewardStateMachine,
        "param_update": ParamUpdateStateMachine,
        "generate": GenerateStateMachine,
        "logp": LogPStateMachine,
        "ref_logp": RefLogPStateMachine,
        "train": TrainStateMachine,
    }
    enhanced_print("create_role_state_machine", None, 
                 f"Creating {role_name} state machine with async RL optimizations (dual buffer + interruptible generation)")
    
    if role_name in state_machines:
        return state_machines[role_name](pipeline, trainer)
    else:
        raise ValueError(f"Unknown role name: {role_name}")
