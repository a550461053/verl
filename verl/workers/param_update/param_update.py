import asyncio
import time
import os
import torch
import ray
import threading
import queue
from enum import Enum, auto
from typing import Optional, Any, Dict, List
from verl.workers.param_update.ray_async_communication import (
    cross_process_ray_put,
    cross_process_ray_get,
)

from verl.trainer.ppo.pipeline.pipeline_utils import enhanced_print
from verl.utils.megatron_utils import per_tensor_generator


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
        target_device: str = "cpu",  # New: target device, default is CPU
        enable_param_sync: bool = True,  # New: whether to enable parameter synchronization
        store_refs_queue = None,  # New: store refs queue
        param_update_preduce_bucket_size_mb: int = 512,  # New: parameter preprocessing bucket size
    ):
        # Basic configuration
        self.model_params = model_params
        self.model_config = model_config
        self.weight_converter = weight_converter
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.target_device = target_device
        self.enable_param_sync = enable_param_sync
        self.store_refs_queue = store_refs_queue
        
        # Async RL optimization switch
        self.enable_async_rl = enable_async_rl
        
        # Log control switch
        self.verbose_logging = os.environ.get('PARAM_UPDATE_VERBOSE_LOG', 'false').lower() == 'true'
        
        # Debug switch: skip func_call execution
        self.skip_func_call = os.environ.get('SKIP_FUNC_CALL', '0') == '1'
        
        # Set two bucket sizes separately
        # 1. Send stage bucket fusion granularity (train stage memory) - can be set larger
        self.send_bucket_size_mb = param_update_preduce_bucket_size_mb
        
        enhanced_print("ParamUpdateManager", None, f"Bucket sizes: send={self.send_bucket_size_mb}MB")
        
        # Initialize parameter metadata
        self._params_meta = []
        
        # Set default Ray collective name
        self.ray_col_name = "actor_rollout_sync"  # Dedicated parameter sync group name, avoid conflict with SGLang communication
        
        # Initialize async RL optimization components
        if self.enable_async_rl:
            self._init_async_rl_components()
        
        self._debug_mode = False  # Debug mode switch
        
        if self.skip_func_call:
            enhanced_print("param_update", None, "⚠️ SKIP_FUNC_CALL enabled - func_call execution will be skipped for debugging")
    
    def setup_for_queue(self, queue):
        self.store_refs_queue = queue

    def _init_async_rl_components(self):
        """Initialize async RL optimization components"""
        # Version counter
        self.current_version = -1
        
        enhanced_print("ParamUpdateManager", None, "Initialized async RL components")

    def get_async_rl_stats(self) -> Dict[str, Any]:
        """Get async RL statistics"""
        if not self.enable_async_rl:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "current_version": self.current_version
        }
    
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
    
    def setup_for_ray_col(self, rank: int, size: int, rank_offset: int, engine_nums: int, engine_tp_size: int, is_engine_master: bool, name: str, backend: str = "nccl"):
        """Setup Ray communication - choose different communication methods based on parameter sync mode"""
        try:
            self.ray_col_name = name
            self.rank = rank
            self.size = size
            self.rank_offset = rank_offset
            self.engine_nums = engine_nums
            self.is_engine_master = is_engine_master
            self.engine_tp_size = engine_tp_size
            
            # Choose communication method based on parameter sync mode
            if self.enable_param_sync:
                # Async mode: use Ray put/get
                enhanced_print("param_update", None, f"Ray put/get communication ready: rank={rank}, size={size}, name={name}")
            else:
                # Sync mode: use Ray Collective
                from ray.util.collective import init_collective_group
                init_collective_group(
                    group_name=name,
                    world_size=size,
                    rank=rank,
                    backend=backend,
                )
                enhanced_print("param_update", None, f"Ray Collective communication ready: rank={rank}, size={size}, name={name}, backend={backend}")
            
            return True
                
        except Exception as e:
            enhanced_print("param_update", None, f"Failed to setup Ray communication: {e}")
            return False
    
    def register_actor_clusters(self, train_ranks: List[int], generate_ranks: List[int], world_size: int):
        """Register train and generate clusters"""
        self.train_ranks = train_ranks
        self.generate_ranks = generate_ranks
        self.world_size = world_size
        
        enhanced_print("param_update", None, f"Registered actor clusters: train_ranks={train_ranks}, generate_ranks={generate_ranks}")
    
    def get_communication_info(self) -> Dict[str, Any]:
        """Get communication information"""
        return {
            "sync_mode": not self.enable_param_sync,
            "train_ranks": getattr(self, 'train_ranks', []),
            "generate_ranks": getattr(self, 'generate_ranks', []),
            "world_size": getattr(self, 'world_size', 0)
        }
    
    def setup_logp_ref_logp_sync(self, logp_rank, ref_logp_rank, size):
        """Setup parameter synchronization for logp and ref_logp workers"""
        if not self.enable_param_sync:
            # Sync mode: use Ray Collective
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
            
            enhanced_print("param_update", None, f"Setup logp/ref_logp sync: logp_group={logp_group_name}, ref_logp_group={ref_logp_group_name}")
        else:
            # Async mode: no additional communication group setup needed
            enhanced_print("param_update", None, "Async mode: no additional communication groups needed for logp/ref_logp")
    
    def get_params_iter(self, target_device="cpu", use_bucketed=False):
        """Get an iterator for the current parameters."""
        # Check if model_params exists (only train nodes have)
        if self.model_params is None:
            enhanced_print("param_update", None, f"get_params_iter: model_params is None, this is not a train node, returning None")
            return None
        
        # Check if bucket optimization is enabled
        bucket_size_mb = self.send_bucket_size_mb
        
        if use_bucketed:
            # Use bucket granularity optimized version with cache
            enhanced_print("param_update", None, f"Using bucketed per_tensor_generator with bucket size {bucket_size_mb}MB")
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
        if hasattr(self, '_bucketed_cache'):
            cache_info = {}
            for key, buckets in self._bucketed_cache.items():
                cache_info[key] = {
                    'bucket_count': len(buckets),
                    'total_tensors': sum(len(bucket) for bucket in buckets)
                }
            return cache_info
        return {}

    def get_params_meta(self):
        # Check if _params_meta is initialized and not empty
        if hasattr(self, '_params_meta') and self._params_meta and len(self._params_meta) > 0:
            return self._params_meta
        
        # Check if model_params is available
        if self.model_params is None:
            enhanced_print("param_update", None, "Warning: model_params is None, cannot get params meta")
            return []
        
        per_tensor_param = self._get_params_iter_bucketed(self.target_device, self.send_bucket_size_mb)

        if per_tensor_param is None:
            enhanced_print("param_update", None, "Error: per_tensor_param returned None")
            return []
        
        # Ensure _params_meta is initialized
        if not hasattr(self, '_params_meta') or self._params_meta is None:
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

        
        enhanced_print("param_update", None, f"Generated {len(self._params_meta)} parameter metadata entries using bucketed generator")
        return self._params_meta

    def set_params_meta(self, params_meta):
        """Set the parameters metadata."""
        self._params_meta = params_meta

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
            enhanced_print("param_update", None, "Warning: StopIteration in consume_per_tensor_generator, generator may be empty")
        except Exception as e:
            # Handle other exceptions
            enhanced_print("param_update", None, f"Error in consume_per_tensor_generator: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def sync_per_tensor_generator(self, func_call=None):
        """Use new async parameter synchronization implementation"""
        return self.async_param_update(func_call)

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
        
        if group_sizes:
            avg_size = sum(group_sizes) / len(group_sizes)
            # enhanced_print("param_update", None, f"Simplified grouping created {len(groups)} groups with Average group size: {avg_size/1024/1024:.1f}MB")
        
        return groups, group_tensor_count

    def _calculate_tensor_size(self, meta):
        """Calculate the size of a tensor."""
        return meta["size"]

    def async_param_update(self, func_call=None, sync_send=False):
        """Async parameter update - start async send and receive"""
        start_time = time.time()
        # enhanced_print("param_update", None, "Starting async parameter update")
        
        # First get groups info, ensure send and recv use same grouping
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

    # Block wait for send to complete
    def wait_for_send_complete(self):
        if hasattr(self, '_async_send_thread') and self._async_send_thread.is_alive():
            self._async_send_thread.join()

    def _start_async_send(self):
        """Start async send thread"""
        if not hasattr(self, '_async_send_thread') or not self._async_send_thread.is_alive():
            self._async_send_thread = threading.Thread(target=self._async_send_worker, daemon=True)
            self._async_send_thread.start()
            # enhanced_print("param_update", None, "Started async send worker thread")

    def _start_async_recv(self, func_call):
        """Start async recv thread"""
        if not hasattr(self, '_async_recv_thread') or not self._async_recv_thread.is_alive():
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
        global_tensor_map, groups, group_tensor_count = self._get_params_iter_bucketed_enhanced(self.target_device, bucket_size_mb=self.send_bucket_size_mb)

        if not self.is_train_master_node():
            # Non-send nodes can exit directly after syncing params
            return

        if self.is_train_master_node() and (not global_tensor_map or not groups):
            enhanced_print("param_update", None, "Async send: no tensor data or groups available")
            return
        
        # Use unified version (each async send advances global version once)
        version = self.current_version + 1
        self.current_version = version
        
        enhanced_print("param_update", None, f"Async send: processing {len(groups)} buckets (version:{version}) with {len(global_tensor_map)} tensors")
        
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
                enhanced_print("param_update", None, f"Async send: {bucket_name} - {found_count}/{len(group)} tensors sent")
        
        t2 = time.time()
        enhanced_print("param_update", None, f"Async send: completed {success_count}/{len(groups)} buckets, cost time:{t2-t1:.2f}, worker thread ending")

    @torch.no_grad()
    def _async_recv_worker(self, func_call):
        """Async recv worker"""
        enhanced_print("param_update", None, "Async recv worker started")
        
        t1 = time.time()
        
        if not self.is_generation_master_node():
            # Non-generator master node, no receive operation
            return True
        
        # Get parameter metadata
        if not self._params_meta:
            self.get_params_meta()
        
        # Use enhanced bucketed function to get grouping info
        _, groups, group_tensor_count = self._get_params_iter_bucketed_enhanced(self.target_device, bucket_size_mb=self.send_bucket_size_mb)
        
        if not groups:
            enhanced_print("param_update", None, "Async recv: no groups available")
            return    
        
        # Initialize current version of receiver (will use version from queue)
        version = self.current_version
        thread_id = threading.get_ident()

        enhanced_print("param_update", None, f"Async recv: processing {len(groups)} buckets ({group_tensor_count} tensors)")
        # Process each bucket, receive in order - ensure same order as send
        success_count = 0
        for i, group in enumerate(groups):
            bucket_name = f"bucket_{i}"
            
            # Get object_refs from store_refs_queue
            store_ref = None
            if self.store_refs_queue is not None:
                # Get object_refs info from queue, use blocking mode
                # enhanced_print("param_update", None, f"i:{i}, rank:{self.rank}, Async recv: waiting for queue_data for {bucket_name}, idx: {self.engine_idx()} queue size: {self.store_refs_queue[self.engine_idx()].qsize()}")
                queue_data = self.store_refs_queue[self.engine_idx()].get()  # 阻塞模式，无timeout
                if self.verbose_logging:
                    enhanced_print("param_update", None, f"Async recv: got queue_data for {queue_data.get('bucket_name') if queue_data else 'None'}, expecting {bucket_name}")
                
                # Verify bucket_name order consistency
                expected_bucket_name = bucket_name
                actual_bucket_name = queue_data.get('bucket_name') if queue_data else None
                
                if actual_bucket_name != expected_bucket_name:
                    assert False, f"ERROR: i:{i}, thread id: {thread_id}, rank:{self.rank}, engine_idx:{self.engine_idx()} Bucket order mismatch! Expected {expected_bucket_name}, got {actual_bucket_name}, exit"
                
                bucket_name = actual_bucket_name
                store_ref = queue_data.get('object_refs') if queue_data else None
                # Get version number for this bucket from queue
                received_version = queue_data.get('version') if queue_data else None
                if received_version is not None:
                    version = received_version
                if self.verbose_logging:
                    enhanced_print("param_update", None, f"i:{i}, thread id: {thread_id}, rank:{self.rank}, engine_idx:{self.engine_idx()}, Async recv: got {len(store_ref)} object_refs for {bucket_name} (version {version})")
            
            # Use _async_recv_bucket_with_store_ref to receive data
            received_tensors = self._async_recv_bucket_with_store_ref(store_ref, version, bucket_name, func_call, group_tensor_count)
            
            if received_tensors:
                success_count += 1
                # enhanced_print("param_update", None, f"i:{i}, rank:{self.rank} Async recv: completed {bucket_name} (version {version})")
            else:
                enhanced_print("param_update", None, f"i:{i}, rank:{self.rank} Async recv: failed {bucket_name} (version {version})")
            
            if self.verbose_logging:
                enhanced_print("param_update", None, f"Async recv: {bucket_name} processed")
        
        t2 = time.time()
        # send / recv run independently, version from send
        self.current_version = version
        enhanced_print("param_update", None, f"Async recv: completed {success_count}/{len(groups)} buckets(version:{version}), cost time:{t2-t1:.2f}, worker thread ending")

    def _async_send_bucket(self, bucket_tensors, ray_rank, bucket_name, version=None):
        """Async send single bucket - train node (using Ray Collective)"""
        if self.verbose_logging:
            enhanced_print("param_update", None, f"Async send: starting {bucket_name} with {len(bucket_tensors)} tensors")
        
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
            if tensor.device.type == 'cuda':
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
                enhanced_print("param_update", None, f"Async send: {group_name} - put={put_time*1000:.1f}ms")
            
            return object_ref
        
        # Send fusion tensors
        object_refs = []  # Store object_refs for cross-ray-actor communication
        
        # Broadcast the whole [(name, tensor)]
        object_refs = broadcast_tensor(fused_tensors, ray_rank, bucket_name)
        
        # Store object_refs to store_refs_queue, for recv thread to use
        if self.store_refs_queue is not None and object_refs:
            # TOOD: Here can be optimized, temporarily use put multiple times;
            # print(f"rank:{self.rank}, len of self.store_refs_queue:{len(self.store_refs_queue)}")
            for queue_idx in range(len(self.store_refs_queue)):
                # enhanced_print("param_update", None, f"rank:{self.rank}, Async send: storing object_refs for {bucket_name}, idx:{queue_idx} queue size={self.store_refs_queue[queue_idx].qsize()}, bucket_name:{bucket_name}")
                self.store_refs_queue[queue_idx].put({
                    'bucket_name': bucket_name,
                    'version': version,
                    'object_refs': object_refs
                })
            # enhanced_print("param_update", None, f"Async send: stored object_refs for {bucket_name} x {len(self.store_refs_queue)} queues")
        
        if self.verbose_logging:
            enhanced_print("param_update", None, f"Async send: {bucket_name} sent successfully")
        
        return True

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
            enhanced_print("param_update", None, f"Async recv: Ray get {len(received_tensors)} tensors completed for {bucket_name} (version {version}) in {get_time:.3f}s")
        
        named_tensors = received_tensors
        
        # Update buffer data
        if func_call and named_tensors:            
            # Directly call func_call to update buffer data
            buffer_success = func_call(named_tensors, version, group_tensor_count)
            
            return buffer_success
        
        return True