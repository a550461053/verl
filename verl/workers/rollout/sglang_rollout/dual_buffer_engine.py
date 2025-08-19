"""
Real dual-buffer SGLang engine implementation

Fix current dual-buffer implementation issues:
1. Support two independent weight copies
2. Update weights without affecting current active buffer
3. Support atomic buffer switching
"""

import asyncio
import os
from typing import List, Tuple, Optional, Dict, Any
import time
import torch
import threading
import pickle

import sglang.srt.entrypoints.engine
from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.entrypoints.engine import UpdateWeightsFromTensorReqInput
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import (
    MultiprocessingSerializer,
)
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)

from verl.trainer.ppo.pipeline.pipeline_utils import enhanced_print


class _BufferManager:
    """Encapsulate buffer index/version/update logic while reusing engine storage.
    This manager operates directly on the owning engine's attributes to avoid
    duplicating state and to keep changes minimal and localized.
    """
    def __init__(self, engine, bucket_size_mb=None):
        self._e = engine
        self._use_reqinput = True  # Enable prefetch, avoid sharing issues by non-FD serialization
        self._use_torch_fd = True  # Keep Torch FD for best performance
        self._use_batch_serialize = True  # Enable batch serialization: fresh serialize the whole list at once when switching
        self._use_reqinput_prefetch = False  # Prefetch serialization in recv stage, can further overlap serialization time; WIP
        
        # Maintain per-buffer (name -> serialized bytes) pool, avoid cross-process FD handle expiration
        # Note: here store the bytes after MultiprocessingSerializer.serialize, not RequestInput object
        self._serialized_pool = [dict(), dict()]
        
        # Unified bucket size setting, support external input or default value
        if bucket_size_mb is not None:
            self._batch_size_mb = bucket_size_mb
        else:
            # Default from environment variable, if not set, use default value
            self._batch_size_mb = int(os.environ.get('PARAM_UPDATE_BUFFER_BUCKET_SIZE_MB', '128'))
        
        self.tp_size = getattr(self._e.server_args, 'tp_size', 1)
        
        # Print configuration information
        enhanced_print("BufferManager", None, f"Initialized with bucket_size_mb={self._batch_size_mb}, tp_size={self.tp_size}")

    def get_bucket_size_mb(self) -> int:
        """Get current bucket size setting"""
        return self._batch_size_mb
    
    def set_bucket_size_mb(self, bucket_size_mb: int):
        """Set bucket size"""
        self._batch_size_mb = bucket_size_mb
        enhanced_print("BufferManager", None, f"Updated bucket_size_mb to {bucket_size_mb}")

    def target_for_update(self) -> int:
        # Default update non-active buffer
        if not self._e._buffer_ready[self._e._active_buffer]:
            return self._e._active_buffer
        return 1 - self._e._active_buffer

    def _ensure_buffer_dict(self, buf_idx: int) -> None:
        # Safety check: ensure index is within valid range
        if buf_idx < 0 or buf_idx >= len(self._e._buffer_weights):
            enhanced_print("DualBufferAsyncEngine", None, f"Invalid buffer index: {buf_idx}, valid range: 0-{len(self._e._buffer_weights)-1}")
            return
        
        if self._e._buffer_weights[buf_idx] is None:
            self._e._buffer_weights[buf_idx] = {}
        
        if self._e._buffer_metas[buf_idx] is None:
            self._e._buffer_metas[buf_idx] = {}

    def _serialize_named_tensors(self, named_tensors):
        # Input is [(name, tensor)]
        # Support batch serialization to improve performance, avoid cross-process sharing issues
        
        if self._use_batch_serialize:
            # Batch serialization: serialize each tensor separately, maintain [(name, tensor)] format
            serialized_list = []
            for name, tensor in named_tensors:
                ser_per_tp = []
                for tp_idx in range(self.tp_size):
                    cpu_tensor = tensor.detach().cpu().contiguous().clone()
                    if self._use_torch_fd:
                        # Create independent tensor copy for each tp, ensure FD token uniqueness
                        serialized = MultiprocessingSerializer.serialize([(name, cpu_tensor)])
                    else:
                        # Use pickle serialization, avoid FD token
                        serialized = pickle.dumps([(name, cpu_tensor)])
                    ser_per_tp.append(serialized)
                serialized_list.append((name, ser_per_tp))
            return serialized_list
        else:
            # Per-tensor serialization (original logic)
            serialized_list = []
            for name, tensor in named_tensors:
                ser_per_tp = []
                cpu_tensor = tensor.detach().cpu().contiguous()
                for tp_idx in range(self.tp_size):
                    if self._use_torch_fd:
                        serialized = MultiprocessingSerializer.serialize([(name, cpu_tensor)])
                    else:   
                        serialized = pickle.dumps([(name, cpu_tensor)])
                    ser_per_tp.append(serialized)
                serialized_list.append((name, ser_per_tp))
            return serialized_list

    def _create_reqinput_from_serialized(self, serialized_items_per_tp):
        # Assemble per-tp bytes into RequestInput
        # serialized_items_per_tp should be list[bytes], each bytes corresponds to one tp's serialized data
        return UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_items_per_tp,
            load_format=None,
            flush_cache=False,
        )

    def _calculate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate tensor size (bytes)"""
        return int(tensor.numel() * tensor.element_size())

    def _create_batches_by_size(self, items, batch_size_mb: int = None):
        """Create batches by size"""
        if batch_size_mb is None:
            batch_size_mb = self._batch_size_mb
        
        batch_bytes_limit = int(batch_size_mb * 1024 * 1024)
        batches = []  # List[List[(name, tensor)]]
        cur_batch = []
        cur_bytes = 0
        
        for name, tensor in items:
            t_bytes = self._calculate_tensor_size(tensor)
            if cur_batch and cur_bytes + t_bytes > batch_bytes_limit:
                batches.append(cur_batch)
                cur_batch = []
                cur_bytes = 0
            cur_batch.append((name, tensor))
            cur_bytes += t_bytes
        
        if cur_batch:
            batches.append(cur_batch)
        
        return batches

    def register_update(self, named_tensors, version: int, group_tensor_count: int) -> bool:
        """Register/accumulate tensors into the non-active buffer with version checks.
        - Reject strictly smaller version to prevent rollback.
        - Allow equal version to accumulate incremental chunks in the same version.
        - Mark buffer as ready on first write for this version.
        - Always call engine.set_version(version) to advance latest_version monotonically.
        Returns whether this update is accepted and applied.
        """
        target_buffer = self.target_for_update()
        existing_version = self._e._buffer_versions[target_buffer]
        if existing_version is not None and version < existing_version:
            enhanced_print("DualBufferAsyncEngine", None, f"Ignore outdated update: buffer {target_buffer} has version {existing_version} > incoming {version}")
            return False

        # Ensure dict initialized and accumulate
        self._ensure_buffer_dict(target_buffer)
        if isinstance(named_tensors, dict):
            iterable = list(named_tensors.items())
        else:
            iterable = list(named_tensors)

        if self._use_batch_serialize:
            # Only save tensor to buffer, no serialization
            for name, tensor in iterable:
                self._e._buffer_weights[target_buffer][name] = tensor.detach().cpu()
                self._e._buffer_metas[target_buffer][name] = 1  # Record metadata, avoid duplicate serialization
            if self._use_reqinput_prefetch:
                # Mark for re-serialization
                self._serialized_pool[target_buffer]["_needs_reserialize"] = True
                # Start unified serialization when last tensor arrives
                if len(self._e._buffer_metas[target_buffer]) == group_tensor_count:
                    self._serialized_pool[target_buffer]["_reqinput"] = self.build_payload_for_apply(target_buffer)
                    self._serialized_pool[target_buffer]["_needs_reserialize"] = False
            else:
                # Mark for re-serialization
                self._serialized_pool[target_buffer]["_needs_reserialize"] = True
        else:
            # Per-tensor mode (original logic)
            serialized_items = self._serialize_named_tensors(iterable)
            for (name, tensor), (s_name, s_list) in zip(iterable, serialized_items):
                assert name == s_name
                self._e._buffer_weights[target_buffer][name] = tensor.detach().cpu()
                # s_list: list[bytes] length == self.tp_size
                self._serialized_pool[target_buffer][name] = s_list

        # Check if all tensors have arrived
        current_tensor_count = len(self._e._buffer_weights[target_buffer])
        if current_tensor_count >= group_tensor_count:
            # All tensors arrived, mark buffer as ready
            self._e._buffer_ready[target_buffer] = True

        self._e._buffer_versions[target_buffer] = version
        # Update latest version via engine API
        self._e.set_version(version)
        # Mark serialized pool as needing refresh

        enhanced_print("DualBufferAsyncEngine", None, f"Registered update: buffer {target_buffer}, version {version}, tensors {len(iterable)}")
        return True

    def get_buffer_items(self, buf_idx: int):
        buf = self._e._buffer_weights[buf_idx]
        if buf is None:
            return []
        return list(buf.items())

    def build_payload_for_apply(self, buf_idx: int):
        """Build payload for sending based on configuration:
        - If use_reqinput=True, merge all serialized bytes of the buffer into a single RequestInput
        - Otherwise return [(name, tensor)] original list
        """
        items = self.get_buffer_items(buf_idx)
        if not items:
            return None

        if not self._serialized_pool[buf_idx]['_needs_reserialize']:
            payload = self._serialized_pool[buf_idx]['_reqinput']
            return payload
        
        if self._use_reqinput:
            if self._use_batch_serialize:
                # Batch mode: split into multiple RequestInputs by batch size, avoid OOM
                batches = self._create_batches_by_size(items)
                
                # Serialize each batch into a RequestInput (fresh, avoid FD reuse)
                req_inputs = []
                for batch in batches:
                    ser_list = []
                    for tp_idx in range(self.tp_size):
                        if self._use_torch_fd:
                            # Build independent [(name, tensor)] list for current tp, avoid extra clone copy, serialization itself allocates unique FD
                            tensors_copy = [(n, t) for n, t in batch]
                            serialized = MultiprocessingSerializer.serialize(tensors_copy)
                        else:
                            serialized = pickle.dumps(batch)
                        ser_list.append(serialized)
                    req_inputs.append(self._create_reqinput_from_serialized(ser_list))
                payload = req_inputs
            else:
                # Per-tensor serialization (immediate fresh serialization, avoid FD reuse)
                req_inputs = []
                for name, _ in items:
                    tensor = self._e._buffer_weights[buf_idx][name]
                    ser_list = []
                    for tp_idx in range(self.tp_size):
                        if self._use_torch_fd:
                            # Clone then immediate serialization for each tp, ensure FD token uniqueness
                            tensor_copy = tensor.clone()
                            serialized = MultiprocessingSerializer.serialize([(name, tensor_copy)])
                        else:
                            serialized = pickle.dumps([(name, tensor)])
                        ser_list.append(serialized)
                    # Don't cache to _serialized_pool, avoid FD cross-use
                    req_inputs.append(self._create_reqinput_from_serialized(ser_list))
                payload = req_inputs
        else:
            payload = items
        
        return payload

    def clear_buffer_metas(self, buf_idx: int):
        """Clear metadata for specified buffer"""
        self._e._buffer_metas[buf_idx].clear()
        
    def get_stats(self) -> dict:
        """Get BufferManager statistics"""
        return {
            "bucket_size_mb": self._batch_size_mb,
            "tp_size": self.tp_size,
            "use_batch_serialize": self._use_batch_serialize,
            "use_torch_fd": self._use_torch_fd,
            "use_reqinput": self._use_reqinput,
            "buffer_ready": self._e._buffer_ready.copy(),
            "buffer_versions": self._e._buffer_versions.copy(),
            "active_buffer": self._e._active_buffer
        }


class DualBufferAsyncEngine:
    """Real dual-buffer SGLang engine, supporting two independent weight copies, inheriting AsyncEngine reuse logic"""
    
    def __new__(cls, **kwargs):
        # Lazy import AsyncEngine to avoid circular reference
        from .sglang_rollout import AsyncEngine
        
        # Dynamically create class inheriting from AsyncEngine
        class DualBufferAsyncEngineImpl(AsyncEngine):
            def __init__(self, **kwargs):
                if 'bucket_size_mb' in kwargs:
                    bucket_size_mb = kwargs.pop('bucket_size_mb', None)

                super().__init__(**kwargs)
                
                # Dual-buffer state
                self._active_buffer = 0  # Current active buffer (0 or 1)
                self._buffer_ready = [False, False]  # Ready state of two buffers
                self._buffer_weights = [None, None]  # Weights of two buffers
                self._buffer_metas = [None, None]
                self._buffer_versions = [None, None]  # Version numbers of two buffers
                self._update_lock = threading.RLock()  # Use threading.RLock instead of asyncio.Lock, support cross-thread
                
                # Version management
                self._current_version = 0  # Current active version
                self._latest_version = 0   # Latest available version
                
                # Default need reload
                self._need_reload = True

                # Buffer manager encapsulating buffer ops
                
                self._bufman = _BufferManager(self, bucket_size_mb=bucket_size_mb)

        def update_weights_from_tensor_sync(self, named_tensors, update_weights_func_call, load_format=None, flush_cache=True, target_buffer=None, version=None):
            """Synchronized version of dual-buffer weight update - thread-safe, support cross-thread call"""
            with self._update_lock:
                # Determine target buffer
                if target_buffer is None:
                    # Default update non-active buffer
                    target_buffer = 1 - self._active_buffer
                
                # Determine version number
                if version is None:
                    version = self._latest_version + 1
                    self._latest_version = version
                
                enhanced_print("DualBufferAsyncEngine", None, f"Updating buffer {target_buffer} (active: {self._active_buffer}) for version {version}")
                
                # Update weights and version of specified buffer
                self._buffer_weights[target_buffer] = named_tensors.copy()
                self._buffer_ready[target_buffer] = True
                self._buffer_versions[target_buffer] = version
                
                enhanced_print("DualBufferAsyncEngine", None, f"Buffer {target_buffer} updated successfully for version {version}")
                
                # Directly call AsyncEngine's update_weights_from_tensor to actually update engine weights
                success = self._apply_weights_to_engine_sync(named_tensors, update_weights_func_call)
                
                if success:
                    # Switch to new updated buffer
                    self._active_buffer = target_buffer
                    self._current_version = version
                    self._bufman.clear_buffer_metas(target_buffer) # Clear buffer reference
                    enhanced_print("DualBufferAsyncEngine", None, f"Switched to buffer {target_buffer} for version {version}")
                else:
                    enhanced_print("DualBufferAsyncEngine", None, f"ERROR: Failed to apply weights to engine for version {version}")
                
                return success
        
        def _register_and_update_buffer(self, target_buffer, named_tensors: Dict[str, torch.Tensor]):
            """Register and update buffer. Note: named_tensors may be Dict[str, Tensor] or Iterable[(name, Tensor)]"""
            # Only create dict when index is not initialized, avoid overwriting accumulated content
            if self._buffer_weights[target_buffer] is None:
                self._buffer_weights[target_buffer] = {}

            # Unified iteration interface, support both dict and (name, tensor) iterable
            if isinstance(named_tensors, dict):
                iterable = named_tensors.items()
            else:
                iterable = named_tensors

            for name, tensor in iterable:
                self._buffer_weights[target_buffer][name] = tensor

        def set_params_meta(self, params_meta):
            self._buffer_meta = params_meta

        def _get_buffer(self, target_buffer):
            # convert dict to [(name, tensor)]
            buffer = self._buffer_weights[target_buffer]
            return [(name, tensor) for name, tensor in buffer.items()]

        def wait_for_buffer_write(self):
            target_buffer = self._active_buffer
            enhanced_print("DualBufferAsyncEngine", None, f"Waiting for buffer {target_buffer} to be ready...")
            while not self._buffer_ready[target_buffer]:
                # wait for update_buffer_data_only update target_buffer;
                time.sleep(0.1)

        def update_buffer_data_only(self, named_tensors, version, group_tensor_count):
            """Only update buffer data, not execute update_weights - used for recv thread
            Rules: by buffer manager: always write to non-active buffer; equal version allows incremental same version; strictly smaller version rejects rollback; and advances latest_version.
            """
            with self._update_lock:
                applied = self._bufman.register_update(named_tensors, version, group_tensor_count)
                return applied

        
        def _run_async_in_sync_context(self, coro):
            """Wrapper function to run async coroutine in sync context"""
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If event loop is running, create new event loop
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                else:
                    # If event loop is not running, use directly
                    result = loop.run_until_complete(coro)
            except RuntimeError:
                # If no event loop, create new
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result = new_loop.run_until_complete(coro)
                new_loop.close()
            return result

        def _apply_weights_to_engine_sync(self, weights, update_weights_func_call):
            """Synchronized version: apply weights to engine - directly call AsyncEngine's update_weights_from_tensor"""
            try:                
                # Directly call AsyncEngine's update_weights_from_tensor method
                # Use sync wrapper to handle async call
                result = self._run_async_in_sync_context(
                    update_weights_func_call(
                        weights,
                        use_reqinput=self._bufman._use_reqinput
                    )
                )
                return result
            except Exception as e:
                print(f"[DualBufferAsyncEngine] ERROR: Failed to apply weights to AsyncEngine: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        def get_current_version(self):
            """Get current active version number"""
            with self._update_lock:
                return self._current_version
        
        def get_latest_version(self):
            """Get latest available version number"""
            with self._update_lock:
                return self._latest_version
        
        def set_version(self, version):
            """Set version number - used for external version management (guarantee monotonic increase), and not rollback current_version"""
            with self._update_lock:
                if version > self._latest_version:
                    self._latest_version = version
                # Avoid rollback current version
                if self._current_version is None or self._current_version < 0:
                    self._current_version = 0
                return True
        
        def execute_update_weights_before_generate(self, update_weights_func_call):
            """Execute weight update before generate - switch to latest version"""
            with self._update_lock:
                enhanced_print("DualBufferAsyncEngine", None, f"Current version: {self._current_version}, Latest version: {self._latest_version}")
                
                # Check if there is a new version available
                if self._latest_version >= self._current_version:
                    # Find buffer containing latest version
                    target_buffer = None
                    for buffer_id in range(2):
                        if self._buffer_versions[buffer_id] == self._latest_version: # and self._buffer_ready[buffer_id]:
                            target_buffer = buffer_id
                            break
                    
                    if target_buffer is not None:                        
                        # Switch to new version
                        success = self.switch_to_buffer_sync(target_buffer, update_weights_func_call)
                        if success:
                            self._current_version = self._latest_version
                            enhanced_print("DualBufferAsyncEngine", None, f"Successfully switched to buffer {target_buffer} for version {self._latest_version}")
                            return True
                        else:
                            enhanced_print("DualBufferAsyncEngine", None, f"Failed to switch to version {self._latest_version}")
                            return False
                    else:
                        enhanced_print("DualBufferAsyncEngine", None, f"Latest version {self._latest_version} not found in any ready buffer")
                        return False
                else:
                    enhanced_print("DualBufferAsyncEngine", None, f"No new version available, current version: {self._current_version}")
                    return True
        
        def switch_to_buffer_sync(self, buffer_id=-1, update_weights_func_call=None):
            """Synchronized version: switch to specified buffer - thread-safe"""
            if buffer_id == -1:
                buffer_id = self.get_latest_version()
            with self._update_lock:
                if not self._buffer_ready[buffer_id]:
                    enhanced_print("DualBufferAsyncEngine", None, f"Buffer {buffer_id} not ready, cannot switch")
                    return False
                
                # First time cannot skip
                # if buffer_id == self._active_buffer:
                #     # Already in use, no need to switch again
                #     return True
                
                # Apply weights of new buffer to engine (only switch active_buffer and current_version after successful application)
                t1 = time.time()
                buffer = self._get_buffer(buffer_id)
                if buffer is None:
                    enhanced_print("DualBufferAsyncEngine", None, f"Buffer {buffer_id} is empty, cannot switch")
                    return False

                payload = self._bufman.build_payload_for_apply(buffer_id)
                t2 = time.time()
                
                if payload is None:
                    enhanced_print("DualBufferAsyncEngine", None, f"Build payload failed for buffer {buffer_id}")
                    return False

                success = self._apply_weights_to_engine_sync(payload, update_weights_func_call)
                
                t3 = time.time()
                # Build payload for buffer 0 took 35.02 s, apply weights took 12.09 s
                enhanced_print("DualBufferAsyncEngine", None, f"Build payload for buffer {buffer_id} took {t2-t1:.2f} s, apply weights took {t3-t2:.2f} s")

                if success:
                    # Switch to new buffer and update current version
                    self._active_buffer = buffer_id
                    self._current_version = self._buffer_versions[buffer_id]
                    self._bufman.clear_buffer_metas(buffer_id)  # Clear buffer reference
                    enhanced_print("DualBufferAsyncEngine", None, f"Successfully applied and switched to buffer {buffer_id} for version {self._current_version}")
                    return True
                else:
                    enhanced_print("DualBufferAsyncEngine", None, f"Failed to apply weights from buffer {buffer_id} to engine; not switching")
                    return False
        
        # Add methods to dynamically created class
        DualBufferAsyncEngineImpl.update_weights_from_tensor_sync = update_weights_from_tensor_sync
        DualBufferAsyncEngineImpl.update_buffer_data_only = update_buffer_data_only
        DualBufferAsyncEngineImpl._apply_weights_to_engine_sync = _apply_weights_to_engine_sync
        DualBufferAsyncEngineImpl.get_current_version = get_current_version
        DualBufferAsyncEngineImpl.get_latest_version = get_latest_version
        DualBufferAsyncEngineImpl.set_version = set_version
        DualBufferAsyncEngineImpl.execute_update_weights_before_generate = execute_update_weights_before_generate
        DualBufferAsyncEngineImpl.switch_to_buffer_sync = switch_to_buffer_sync
        DualBufferAsyncEngineImpl._run_async_in_sync_context = _run_async_in_sync_context
        DualBufferAsyncEngineImpl._register_and_update_buffer = _register_and_update_buffer
        DualBufferAsyncEngineImpl._get_buffer = _get_buffer
        DualBufferAsyncEngineImpl.wait_for_buffer_write = wait_for_buffer_write
        DualBufferAsyncEngineImpl.set_params_meta = set_params_meta
        
        # Add statistics and configuration methods
        def get_stats(self) -> dict:
            """Get dual-buffer engine statistics"""
            return {
                "active_buffer": self._active_buffer,
                "buffer_ready": self._buffer_ready.copy(),
                "buffer_versions": self._buffer_versions.copy(),
                "current_version": self._current_version,
                "latest_version": self._latest_version,
                "need_reload": self._need_reload,
                "buffer_manager": self._bufman.get_stats()
            }
        
        def get_bucket_size_mb(self) -> int:
            """Get current bucket size setting"""
            return self._bufman.get_bucket_size_mb()
        
        def set_bucket_size_mb(self, bucket_size_mb: int):
            """Set bucket size"""
            self._bufman.set_bucket_size_mb(bucket_size_mb)
        
        DualBufferAsyncEngineImpl.get_stats = get_stats
        DualBufferAsyncEngineImpl.get_bucket_size_mb = get_bucket_size_mb
        DualBufferAsyncEngineImpl.set_bucket_size_mb = set_bucket_size_mb
        
        # Return new instance
        return DualBufferAsyncEngineImpl(**kwargs)
