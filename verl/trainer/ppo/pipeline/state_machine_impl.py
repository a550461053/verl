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
from .utils import resource_lock, engine_resource_lock, dataloader_scheduler_lock, global_timing_collector, timing_decorator

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
    Role,
    OmegaConf,
    create_colocated_worker_cls,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    marked_timer,
    compute_advantage,
    compute_response_mask,
    agg_loss,
    AdvantageEstimator,
    DataProto,
    apply_kl_penalty,
)



# Ray remote functions for async computation
@ray.remote(num_cpus=1)
def compute_logp_async(actor_wg, batch: DataProto):
    """Async logp computation using ray.remote"""
    # Note: Resource lock is handled in the calling state machine
    return actor_wg.compute_log_prob(batch)

@ray.remote(num_cpus=1)
def compute_ref_logp_async(ref_policy_wg, actor_wg, batch: DataProto, ref_in_actor: bool):
    """Async ref_logp computation using ray.remote"""
    # Note: Resource lock is handled in the calling state machine
    if not ref_in_actor:
        return ref_policy_wg.compute_ref_log_prob(batch)
    else:
        return actor_wg.compute_ref_log_prob(batch)

@ray.remote(num_cpus=1)
def compute_reward_async(batch: DataProto, reward_fn):
    """Async reward computation using ray.remote"""
    # Note: Resource lock is handled in the calling state machine
    return compute_reward(batch, reward_fn)


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
            # Acquire dataloader scheduler lock before loading data
            await dataloader_scheduler_lock.acquire("train_dataloader")
            enhanced_print("dataloader", None, "Acquired dataloader scheduler lock for training data")
            
            try:
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
            finally:
                # Release dataloader scheduler lock after loading data
                await dataloader_scheduler_lock.release("train_dataloader")
                enhanced_print("dataloader", None, "Released dataloader scheduler lock for training data")
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
    
    async def get_input_data(self) -> Optional[Any]:
        """Block getting dataloader; wait for generate data to arrive"""

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
        
        with marked_timer("step", timing_raw):
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
            enhanced_print("rollout", None, f"Pushing to logp for step {data['step']}")
            push_tasks.append(self.pipeline.push("rollout", "logp", (data["step"], batch)))
        
        if self.pipeline.is_in_pipeline("ref_logp"):
            enhanced_print("rollout", None, f"Pushing to ref_logp for step {data['step']}")
            push_tasks.append(self.pipeline.push("rollout", "ref_logp", (data["step"], batch)))
        
        if self.pipeline.is_in_pipeline("reward"):
            enhanced_print("rollout", None, f"Pushing to reward for step {data['step']}")
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
        
        # Wait for validation result if validation was triggered
        validation_result = None
        if "validation" in self.pipeline.role:
            # Check if validation was actually performed (non-blocking)
            validation_result = await self.pipeline.pull("validation", "train")
            # If validation_result is None, it means validation was skipped for this step
            # If validation_result is ("INITIAL_VALIDATION", val_metrics), it's initial validation

        # Check if all data is received
        if logp_result is None or ref_logp_result is None or reward_result is None:
            return None
            
        logp_step, old_log_prob = logp_result
        ref_logp_step, ref_log_prob = ref_logp_result
        reward_step, reward_tensor, reward_extra_infos_dict = reward_result
        
        # Verify step consistency
        assert logp_step == ref_logp_step == reward_step, f"Step mismatch: logp_step={logp_step}, ref_logp_step={ref_logp_step}, reward_step={reward_step}"
        
        # Process validation result if available
        if validation_result is not None:
            if len(validation_result) == 2:  # Actual validation results
                val_step, val_metrics = validation_result
                if val_step == "INITIAL_VALIDATION":
                    enhanced_print("train", None, "Received initial validation results")
                    # For initial validation, log directly
                    if self.logger and val_metrics:
                        self.logger.log(data=val_metrics, step=0)  # Log at step 0 for initial validation
                else:
                    enhanced_print("train", None, f"Received validation results for step {val_step}")
                    # Store validation metrics for later logging with training metrics
                    self.last_val_metrics = val_metrics
            # If validation_result is None, validation was skipped for this step
        
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
        
        with marked_timer("step", timing_raw):
            # Acquire resource lock (for training phase)
            await resource_lock.acquire("train", self.trainer.global_steps)

            # recompute old_log_probs
            with marked_timer("old_log_prob", timing_raw):
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
                with marked_timer("ref", timing_raw):
                    batch = batch.union(ref_log_prob)

            # compute values; TODO support async-rl
            if self.trainer.use_critic:
                with marked_timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            with marked_timer("adv", timing_raw):
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
                    config=self.trainer.config.algorithm,
                )

            # update critic
            if self.trainer.use_critic:
                with marked_timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.trainer.config.trainer.critic_warmup <= self.trainer.global_steps:
                # update actor
                with marked_timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.trainer.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.trainer.actor_wg.update_actor(batch)

                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.trainer.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with marked_timer("dump_rollout_generations", timing_raw):
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
                with marked_timer("save_checkpoint", timing_raw):
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

        # Add validation metrics if available (following the pattern from other trainers)
        if self.last_val_metrics is not None:
            metrics.update(self.last_val_metrics)
            enhanced_print("train", None, f"Added validation metrics to training metrics for step {self.trainer.global_steps}")

        self.logger.log(data=metrics, step=self.trainer.global_steps)
        
        # Clear validation metrics after logging to avoid duplicate logging
        if self.last_val_metrics is not None:
            self.last_val_metrics = None

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
        
        # 1. First, trigger param_update to ensure it gets resource lock first
        await self.pipeline.push(src_role="train", dst_role="param_update", data=global_steps)
        enhanced_print("train", None, f"Sent training completion signal for step {global_steps} to param_update")

        # 2. Trigger validation (if validation state machine exists)
        if "validation" in self.pipeline.role:
            await self.pipeline.push(src_role="train", dst_role="validation", data=global_steps)
            enhanced_print("train", None, f"Sent validation trigger for step {global_steps} to validation")

        # 3. Trigger logp/ref_logp for next step (these will wait for param_update signal)
        if not self.trainer.config.trainer.get("sperated_train_logp", False):
            await self.pipeline.push(src_role="train", dst_role="logp", data=global_steps)
        if not self.trainer.sperated_ref_model:
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
        """Process reward calculation logic - async execution using to_thread"""
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
        
        # Use asyncio.to_thread for async computation (like generate)
        enhanced_print("reward", None, f"Starting async reward computation using to_thread for step {step}")
        
        if self.trainer.use_rm:
            # RM computation (if needed)
            reward_tensor = self.trainer.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)
        
        # Launch async reward computation using to_thread
        reward_tensor, reward_extra_infos_dict = await asyncio.to_thread(compute_reward, batch, self.trainer.reward_fn)
        enhanced_print("reward", None, f"Async reward computation completed for step {step}")
        
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
            
            # Always wait for train to complete before logp to ensure proper resource lock order
            # This ensures: train -> param_update -> logp sequence
            enhanced_print("logp", None, f"Waiting for train to complete")
            await self.pipeline.pull("train", "logp")
            enhanced_print("logp", None, f"Train step completed, continuing with logp")
            
            # Wait for param_update to complete before logp (unless separated deployment)
            if not self.trainer.config.trainer.get("sperated_train_logp", False):
                enhanced_print("logp", None, f"Waiting for param_update to complete")
                await self.pipeline.pull("param_update", "logp")
                enhanced_print("logp", None, f"Param_update completed, continuing with logp")
            return batch
        except Exception as ex:
            enhanced_print("logp", None, f"Error in get_input_data: {ex}")
            return None
    
    @timing_decorator("logp")
    async def process_data(self, data: Any) -> Any:
        """Process LogP calculation logic - async execution"""
        if data is None:
            enhanced_print("logp", None, "Received None data, waiting...")
            return None
        
        if data == "END":
            return "END"
        step, batch = data
        if batch == PIPELINE_END_SIGNAL:
            return "END"
        enhanced_print("logp", None, f"Computing logp for step {step}")
        
        # Acquire resource lock before launching async task
        await resource_lock.acquire("logp", step)
        
        # Use asyncio.to_thread for async computation (like ref_logp and reward)
        enhanced_print("logp", None, f"Starting async LogP computation using to_thread for step {step}")
        
        def compute_logp_sync():
            return self.trainer.actor_wg.compute_log_prob(batch)
        
        # Launch async computation using to_thread
        old_log_prob = await asyncio.to_thread(compute_logp_sync)
        enhanced_print("logp", None, f"Async LogP computation completed for step {step}")
        
        # Release resource lock after computation is complete
        await resource_lock.release("logp")
        
        # Return result directly (no future needed)
        return (step, old_log_prob)
    
    async def send_output_data(self, data: Any) -> bool:
        """Send LogP results - block to ensure data transfer"""
        if data is None:
            return False
        if data == "END":
            # no need to send to train queue
            return True
        
        step, old_log_prob = data
        
        # Block send results, ensure train can receive data
        await self.pipeline.push("logp", "train", (step, old_log_prob))
        enhanced_print("logp", None, f"Sent logp result for step {step}")

        if not self.trainer.sperated_ref_model:
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
        self.sperated_ref_model = self.trainer.sperated_ref_model
        # set offpolicy steps: generate ahead param_update
        self.generate_ahead_steps = trainer.config.trainer.get("generate_ahead_steps", 3)
        enhanced_print("ref_logp", None, f"Using sperated_ref_model: {self.sperated_ref_model}")
        
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
            if not self.sperated_ref_model:
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
        """Process reference LogP calculation logic - async execution using to_thread"""
        if data is None:
            enhanced_print("ref_logp", None, "Received None data, waiting...")
            return None
        
        if data == "END":
            return "END"
        
        step, batch = data
        if batch == PIPELINE_END_SIGNAL:
            return "END"
        enhanced_print("ref_logp", None, f"Computing ref_logp for step {step}")

        # Acquire resource lock if not separated (for shared resources)
        if not self.sperated_ref_model:
            await resource_lock.acquire("ref_logp", step)
            enhanced_print("ref_logp", None, f"Acquired resource lock for step {step}")

        # Use asyncio.to_thread for async computation (like generate)
        enhanced_print("ref_logp", None, f"Starting async Ref LogP computation using to_thread for step {step}")
        
        def compute_ref_logp_sync():
            if not self.trainer.ref_in_actor:
                return self.trainer.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                return self.trainer.actor_wg.compute_ref_log_prob(batch)
        
        # Launch async computation using to_thread
        ref_log_prob = await asyncio.to_thread(compute_ref_logp_sync)
        enhanced_print("ref_logp", None, f"Async Ref LogP computation completed for step {step}")
        
        # Release resource lock if not separated
        if not self.sperated_ref_model:
            await resource_lock.release("ref_logp")
            enhanced_print("ref_logp", None, f"Released resource lock for step {step}")
        
        return (step, ref_log_prob)
    
    async def send_output_data(self, data: Any) -> bool:
        """Send reference LogP results - block to ensure data transfer"""
        if data is None:
            return False
        if data == "END":
            # no need to send to train queue
            return True
        
        step, ref_log_prob = data
        
        # Block send results, ensure train can receive data
        await self.pipeline.push("ref_logp", "train", (step, ref_log_prob))
        enhanced_print("ref_logp", None, f"Sent ref_logp result for step {step}")
        return True


class ParamUpdateStateMachine(BaseRoleStateMachine):
    """asyncRLparameterupdatestate machine"""
    
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
        
        rollout_wg = self.trainer.rollout_wg
        actor_wg = self.trainer.actor_wg
        self.has_param_update_manager = hasattr(actor_wg, 'async_param_update')
        
        if self.has_param_update_manager:
            enhanced_print("AsyncRLParamUpdate", None, "Using param_update_manager for async parameter synchronization")
        else:
            enhanced_print("AsyncRLParamUpdate", None, "param_update_manager not available, falling back to sync update")
    
    async def get_input_data(self) -> Optional[Any]:
        """getting parameter update"""
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
        """Process parameter update"""
        if data == "END":
            return "END"
        elif data is None:
            enhanced_print("param_update", None, "Received None data, waiting...")
            return None
        
        global_steps = data

        enhanced_print("param_update", None, f"Starting param update for step {global_steps}")
        
        start_time = time.time()

        # For NCCL sync mode, use await instead of create_task to ensure blocking execution
        enable_param_async = getattr(self.trainer.config.actor_rollout_ref.rollout, 'enable_param_async', False)
        if not enable_param_async:
            # NCCL sync mode: use await for blocking execution
            await engine_resource_lock.acquire("param_update")
            await resource_lock.acquire("param_update", global_steps)
            send_success = await self._perform_async_param_update_send_phase(global_steps)
            recv_success = await self._perform_async_param_update_recv_phase(global_steps)
            await resource_lock.release("param_update", global_steps)
            await engine_resource_lock.release("param_update")
            param_update_task = ("sync", send_success, recv_success)  # Add mode indicator
        else:
            # CPU async mode: use create_task for async execution
            send_task = asyncio.create_task(
                self._perform_async_param_update_send_phase(global_steps)
            )
            recv_task = asyncio.create_task(
                self._perform_async_param_update_recv_phase(global_steps)
            )
            param_update_task = ("async", send_task, recv_task)  # Add mode indicator

        self.stats["async_updates"] += 1
        enhanced_print("param_update", None, f"Param update async:{enable_param_async} tasks created for step {global_steps}")

        task_creation_time = time.time() - start_time

        enhanced_print("param_update", None, 
                        f"Param update task created for step {global_steps} in {task_creation_time:.3f}s")
        
        return (global_steps, param_update_task)

    async def _perform_async_param_update_send_phase(self, global_steps: int, resource_lock_enable=True) -> bool:
        """async parameter update - send phase only"""
        enhanced_print("param_update", None, f"Background async param update send phase started for step {global_steps}")

        if resource_lock_enable:
            await resource_lock.acquire("param_update", global_steps)

        start_time = time.time()

        self.trainer.actor_wg.async_param_update()
        self.trainer.rollout_wg.async_param_update()
        
        # waiting send completed
        self.trainer.actor_wg.wait_for_send_complete()

        send_time = time.time() - start_time

        self.stats["updates"] += 1
        self.stats["total_time"] += send_time
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["updates"]
        self.stats["min_time"] = min(self.stats["min_time"], send_time)
        self.stats["max_time"] = max(self.stats["max_time"], send_time)
        
        enhanced_print("param_update", None, 
                        f"Background async param update send phase completed for step {global_steps} in {send_time:.3f}s")
        
        if resource_lock_enable:
            await resource_lock.release("param_update", global_steps)

        return True

    async def _perform_async_param_update_recv_phase(self, global_steps: int) -> bool:
        """async parameter update - recv phase only"""
        enhanced_print("param_update", None, f"Background async param update recv phase started for step {global_steps}")

        # waiting recv completed
        self.trainer.rollout_wg.wait_for_recv_complete()
        
        enhanced_print("param_update", None, 
                        f"Background async param update recv phase completed for step {global_steps}")

        return True

    async def _perform_async_param_update_background(self, global_steps: int) -> bool:
        """async parameter update - complete process (for backward compatibility)"""
        # Send phase
        send_success = await self._perform_async_param_update_send_phase(global_steps)
        if not send_success:
            return False
        
        # Recv phase
        recv_success = await self._perform_async_param_update_recv_phase(global_steps)
        return recv_success

    async def send_output_data(self, data: Any) -> bool:
        """send update completed with optimized dependencies"""
        if data == "END":
            enhanced_print("param_update", None, "Async param update completed, END signal processed")
            await self.pipeline.push("param_update", "generate", PIPELINE_END_SIGNAL)
            return True
        elif data is None:
            return False
        
        if isinstance(data, tuple) and len(data) == 2:
            global_steps, param_update_task = data
            enhanced_print("param_update", None, f"Processing param update tasks for step {global_steps}")

            # Handle separate send/recv tasks for async updates
            if isinstance(param_update_task, tuple) and len(param_update_task) >= 2:
                mode = param_update_task[0]
                if mode == "sync":
                    # NCCL sync mode: param_update_task is (mode, send_success, recv_success)
                    _, send_success, recv_success = param_update_task
                    if send_success:
                        enhanced_print("param_update", None, f"Sync param update send phase completed for step {global_steps}")
                        
                        # Send signal to logp (only needs send completion)
                        if not self.trainer.config.trainer.get("sperated_train_logp", False):
                            await self.pipeline.push("param_update", "logp", global_steps)
                            enhanced_print("param_update", None, f"Sent completion signal to logp for step {global_steps} (after send complete)")
                    
                    # In NCCL sync mode, recv is already completed, so we can directly proceed
                    if recv_success:
                        enhanced_print("param_update", None, f"Sync param update recv phase completed for step {global_steps}")
                        
                        # Send signal to generate (needs recv completion)
                        await self.pipeline.push("param_update", "generate", global_steps)
                        enhanced_print("param_update", None, f"Sent completion signal to generate for step {global_steps} (after recv complete)")
                else:
                    # CPU async mode: param_update_task is (mode, send_task, recv_task)
                    _, send_task, recv_task = param_update_task
                    
                    # Wait for send phase completion (for logp)
                    send_success = await send_task
                    if send_success:
                        enhanced_print("param_update", None, f"Background param update send phase completed for step {global_steps}")
                        
                        # Send signal to logp (only needs send completion)
                        if not self.trainer.config.trainer.get("sperated_train_logp", False):
                            await self.pipeline.push("param_update", "logp", global_steps)
                            enhanced_print("param_update", None, f"Sent completion signal to logp for step {global_steps} (after send complete)")
                
                    # Wait for recv phase completion (for generate)
                    recv_success = await recv_task
                    if recv_success:
                        enhanced_print("param_update", None, f"Background param update recv phase completed for step {global_steps}")
                        
                        # Send signal to generate (needs recv completion)
                        await self.pipeline.push("param_update", "generate", global_steps)
                        enhanced_print("param_update", None, f"Sent completion signal to generate for step {global_steps} (after recv complete)")
            else:
                # Handle single task for sync updates
                success = await param_update_task
                if success:
                    enhanced_print("param_update", None, f"Background param update completed for step {global_steps}")
                    
                    # Send signals to both logp and generate (sync update completes both phases)
                    if not self.trainer.config.trainer.get("sperated_train_logp", False):
                        await self.pipeline.push("param_update", "logp", global_steps)
                        enhanced_print("param_update", None, f"Sent completion signal to logp for step {global_steps}")
                    
                    await self.pipeline.push("param_update", "generate", global_steps)
                    enhanced_print("param_update", None, f"Sent completion signal to generate for step {global_steps}")
                else:
                    enhanced_print("param_update", None, f"Background param update failed for step {global_steps}")
                    return False
        else:
            global_steps = data
        
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
    """asyncRLgenerationstate machine"""
    
    def __init__(self, pipeline, trainer):
        super().__init__("generate", pipeline)
        self.trainer = trainer
        self.first_generation = True
        
        # set offpolicy steps: generate ahead param_update
        # +1 because the first generation is not offpolicy
        self.generate_ahead_steps = trainer.config.trainer.get("generate_ahead_steps", 2) + 1
        self.last_param_update_step = 0

        enhanced_print("generate", None, f"Configured generate ahead: {self.generate_ahead_steps} steps")

    @timing_decorator("generate")
    async def process_data(self, data: Any) -> Any:
        """Processgenerationlogic"""
        if data == "END":
            return "END"
        elif data is None:
            enhanced_print("generate", None, "Received None data, waiting...")
            return None
        
        step, gen_batch = data
        enhanced_print("generate", None, f"Starting generation task for step {step}")
        
        # Acquire engine resource lock before starting generation
        await engine_resource_lock.acquire("generate")
        enhanced_print("generate", None, f"Acquired engine resource lock for step {step}")
        
        generation_task = asyncio.create_task(
            asyncio.to_thread(self._generate_sync, gen_batch, step)
        )
        
        # updategenerate_global_step
        self.trainer.generate_global_step += 1
        
        enhanced_print("generate", None, f"Generation task created for step {step}")
        
        return (step, generation_task)

    async def send_output_data(self, data: Any) -> bool:
        """sendgenerationresults"""
        if data == "END":
            enhanced_print("generate", None, "Sending END signal to rollout")
            await self.pipeline.push("generate", "rollout", PIPELINE_END_SIGNAL)
            return True
        elif data is None:
            return False
        
        if isinstance(data, tuple) and len(data) == 2:
            step, generation_task = data
            enhanced_print("generate", None, f"Waiting for background generation task to complete for step {step}")
            
            gen_batch_output = await generation_task
            if gen_batch_output is None:
                raise Exception(f"Generation failed for step {step}")

            # Release engine resource lock after generation is complete
            await engine_resource_lock.release("generate")
            enhanced_print("generate", None, f"Released engine resource lock for step {step}")

        else:
            step, gen_batch_output = data
        
        enhanced_print("generate", None, f"Sending generation result to rollout for step {step}")
        await self.pipeline.push("generate", "rollout", (step, gen_batch_output))
        enhanced_print("generate", None, f"Generation result sent to rollout for step {step}")
        
        return True
    
    async def get_input_data(self) -> Optional[Any]:
        """gettinggenerationdata"""        
        # first generation need waiting param_update completed
        if self.first_generation:
            enhanced_print("generate", None, "First generation, waiting for initial param_update to complete...")
            param_update_signal = await self.pipeline.pull("param_update", "generate")
            
            if param_update_signal == PIPELINE_END_SIGNAL:
                enhanced_print("generate", None, "Received END signal from param_update")
                return "END"
            elif param_update_signal is None:
                return None
            
            self.last_param_update_step = param_update_signal
            enhanced_print("generate", None, f"First generation: received param_update completion signal for step {param_update_signal}")
            self.first_generation = False
        
        # waiting dataloader
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

        # Check the distance between generate and param_update. 
        # If it is too far, Blocking to waiting param_update.
        while step > self.last_param_update_step + self.generate_ahead_steps:
            enhanced_print("generate", None, f"Step {step} is too far ahead of param_update {self.last_param_update_step}, waiting for next param_update...")
            param_update_signal = await self.pipeline.pull("param_update", "generate")
            
            if param_update_signal == PIPELINE_END_SIGNAL:
                enhanced_print("generate", None, "Received END signal from param_update while waiting")
                return "END"
            elif param_update_signal is None:
                return None
            
            self.last_param_update_step = param_update_signal
            enhanced_print("generate", None, f"Updated param_update step to {param_update_signal}")
        
        enhanced_print("generate", None, f"Got generation task for step {step}")
        return (step, gen_batch)
    
    def _generate_sync(self, gen_batch, step: int):
        """synchronization generation"""
        enhanced_print("generate", None, f"Background generation started for step {step}")
        
        start_time = time.time()
        
        # executegeneration
        wg = self.trainer.rollout_wg
        gen_batch_output = wg.generate_sequences_sperated(gen_batch)
        
        generation_time = time.time() - start_time
        
        enhanced_print("generate", None, f"Background generation completed for step {step} in {generation_time:.3f}s")
        
        return gen_batch_output


    def get_status_info(self) -> Dict[str, Any]:
        """getting detailed status information"""
        async_rl_stats = {}
        return {
            "type": "async_rl_generate",
            "has_async_rl_support": True,
            "async_rl_stats": async_rl_stats,
            "description": "Async RL generate with interruptible generation"
        }


class ValidationStateMachine(BaseRoleStateMachine):
    """Validation state machine for periodic validation during training"""
    # input: 
    #   train -> validation (validation trigger)
    # output:
    #   validation -> train (validation results)
    
    def __init__(self, pipeline, trainer):
        super().__init__("validation", pipeline)
        self.trainer = trainer
        self.validation_freq = self.trainer.config.trainer.get("test_freq", 100)
        self.last_validation_step = 0
        self.initial_validation_done = False
        self.val_before_train = self.trainer.config.trainer.get("val_before_train", True)
        self.val_only = self.trainer.config.trainer.get("val_only", False)
        enhanced_print("validation", None, f"Validation frequency: {self.validation_freq}, val_before_train: {self.val_before_train}")
    
    async def get_input_data(self) -> Optional[Any]:
        """Get validation trigger from train or perform initial validation"""
        try:
            # Check if we need to perform initial validation before training
            if not self.initial_validation_done and self.val_before_train and self.trainer.val_reward_fn is not None:
                enhanced_print("validation", None, "Performing initial validation before training")
                self.initial_validation_done = True
                return "INITIAL_VALIDATION"

            if not self.val_before_train and not self.initial_validation_done:
                # Skip initial validation if not required
                self.initial_validation_done = True
                await self.pipeline.push("validation", "train", None)
                return None

            # Block waiting for validation trigger from train
            trigger = await self.pipeline.pull("train", "validation")
            
            if trigger is None:
                return None
            
            if trigger == PIPELINE_END_SIGNAL:
                return "END"
            
            # Check if validation should be performed
            current_step = trigger
            is_last_step = current_step >= self.trainer.total_training_steps
            if (current_step - self.last_validation_step >= self.validation_freq) or is_last_step:
                self.last_validation_step = current_step
                enhanced_print("validation", None, f"Validation triggered for step {current_step}")
                return current_step
            else:
                enhanced_print("validation", None, f"Validation skipped for step {current_step}, next at {self.last_validation_step + self.validation_freq}")
                # Send None to indicate validation was skipped
                await self.pipeline.push("validation", "train", None)
                return None
                
        except Exception as ex:
            enhanced_print("validation", None, f"Error in get_input_data: {ex}")
            return None
    
    @timing_decorator("validation")
    async def process_data(self, data: Any) -> Any:
        """Process validation logic using to_thread"""
        if data is None:
            return None
        
        if data == "END":
            return "END"
        
        if data == "INITIAL_VALIDATION":
            enhanced_print("validation", None, "Starting initial validation before training")
            
            # Acquire both dataloader scheduler lock and engine resource lock for validation
            await dataloader_scheduler_lock.acquire("validation_dataloader")
            await engine_resource_lock.acquire("validation")
            enhanced_print("validation", None, "Acquired dataloader scheduler lock and engine resource lock for validation")
            
            try:
                # Use asyncio.to_thread for validation computation
                def validate_sync():
                    return self.trainer._validate()
                val_metrics = await asyncio.to_thread(validate_sync)
                
                enhanced_print("validation", None, "Async initial validation computation completed")
                
                # Check if val_only mode is enabled
                if self.val_only:
                    enhanced_print("validation", None, "val_only mode enabled, stopping after initial validation")
                    return "VAL_ONLY_END"
                
                return ("INITIAL_VALIDATION", val_metrics)
            finally:
                # Release both locks after validation
                await engine_resource_lock.release("validation")
                await dataloader_scheduler_lock.release("validation_dataloader")
                enhanced_print("validation", None, "Released dataloader scheduler lock and engine resource lock for validation")
        
        step = data
        enhanced_print("validation", None, f"Starting validation for step {step}")
        
        # Acquire both dataloader scheduler lock and engine resource lock for validation
        await dataloader_scheduler_lock.acquire("validation_dataloader")
        await engine_resource_lock.acquire("validation")
        enhanced_print("validation", None, f"Acquired dataloader scheduler lock and engine resource lock for validation step {step}")
        
        try:
            # Use asyncio.to_thread for validation computation
            def validate_sync():
                return self.trainer._validate()
            val_metrics = await asyncio.to_thread(validate_sync)
            
            enhanced_print("validation", None, f"Async validation computation completed for step {step}")
            
            return (step, val_metrics)
        finally:
            # Release both locks after validation
            await engine_resource_lock.release("validation")
            await dataloader_scheduler_lock.release("validation_dataloader")
            enhanced_print("validation", None, f"Released dataloader scheduler lock and engine resource lock for validation step {step}")
    
    async def send_output_data(self, data: Any) -> bool:
        """Send validation results back to train"""
        if data is None:
            return False
        if data == "END":
            return True
        if data == "VAL_ONLY_END":
            # Send end signal to all state machines
            for role_name in ["dataloader", "rollout", "train", "generate", "reward", "logp", "ref_logp", "param_update"]:
                if role_name in self.pipeline.role:
                    await self.pipeline.push("validation", role_name, PIPELINE_END_SIGNAL)
            enhanced_print("validation", None, "Sent end signal to all state machines due to val_only mode")
            return True
        
        step, val_metrics = data
        
        # Handle initial validation results
        if step == "INITIAL_VALIDATION":
            # For initial validation, log directly and send to train
            enhanced_print("validation", None, f"Initial validation metrics: {val_metrics}")
            if self.trainer.config.trainer.get("val_only", False):
                # In val_only mode, just log and end
                return True
            else:
                # Send initial validation results to train
                await self.pipeline.push("validation", "train", (step, val_metrics))
                enhanced_print("validation", None, "Sent initial validation results to train")
                return True
        
        # Send validation results back to train
        await self.pipeline.push("validation", "train", (step, val_metrics))
        enhanced_print("validation", None, f"Sent validation results for step {step}")
        return True


def create_role_state_machine(role_name: str, pipeline, trainer, use_async_rl: bool = False) -> BaseRoleStateMachine:
    """
    Create a role state machine factory function
    
    Args:
        role_name: role name, e.g., "dataloader", "rollout", "reward", "param_update", "generate", "logp", "ref_logp", "train"
        pipeline: pipeline instance
        trainer: trainer instance
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
        "validation": ValidationStateMachine,
    }
    enhanced_print("create_role_state_machine", None, 
                 f"Creating {role_name} state machine with async RL optimizations (dual buffer + interruptible generation)")
    
    if role_name in state_machines:
        return state_machines[role_name](pipeline, trainer)
    else:
        raise ValueError(f"Unknown role name: {role_name}")
