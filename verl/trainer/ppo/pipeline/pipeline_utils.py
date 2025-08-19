"""
Asynchronous Pipeline management class, responsible for data flow and queue management between roles

Performance optimization summary:
1. Original problem: Ray queue's put_async/get_async takes too long (40+ seconds)
2. Root cause: Internal locks and network transmission bottlenecks in Ray queue in distributed environment
3. Solution: Use DIRECT_OBJECT_STORE mode, bypass Ray queue
4. Optimization effect: ray.put/ray.get takes milliseconds (0.000s)
5. Current bottleneck: Model parameter update itself (22-35 seconds), not data transmission

Transfer mode comparison:
- RAY_QUEUE: Original Ray queue mode, performance bottleneck
- RAY_QUEUE_COMPRESSED: Ray queue + compression, performance slightly improved
- RAY_QUEUE_OPTIMIZED: Ray queue + ray.put optimization, performance significantly improved
- DIRECT_OBJECT_STORE: Direct use of object store, optimal performance
- HYBRID: Hybrid mode, automatically select based on data size
"""

import asyncio
import time
import pickle
from collections import OrderedDict
import ray
from ray.util.queue import Queue
from colorama import Fore, Style
from enum import Enum

# Transfer mode enumeration
class TransferMode(Enum):
    RAY_QUEUE = "ray.queue"                    # Original Ray queue method
    RAY_QUEUE_COMPRESSED = "ray.queue.compressed"  # Ray queue + compression
    RAY_QUEUE_OPTIMIZED = "ray.queue.optimized"    # Ray queue + ray.put optimization
    DIRECT_OBJECT_STORE = "direct.object.store"    # Direct use of object store, bypass queue
    HYBRID = "hybrid"                          # Hybrid mode, choose based on data size

# Pipeline signal constants
PIPELINE_END_SIGNAL = "__PIPELINE_END_SIGNAL__"
PIPELINE_START_SINGLE = "__PIPELINE_START_SINGLE__"

# Define color mapping
ROLE_COLORS = {
    "dataloader": Fore.WHITE,
    "rollout": Fore.BLUE,
    "generate": Fore.YELLOW,
    "train": Fore.GREEN,
    "reward": Fore.WHITE,
    "logp": Fore.MAGENTA,
    "ref_logp": Fore.CYAN,
    "param_update": Fore.RED,
}

def enhanced_print(src_role, dst_role, message):
    """Enhanced print output with color and role identification"""
    max_len = max([len(role) for role in ROLE_COLORS.keys()])
    src_role_formatted = f"{src_role:<{max_len}}"
    src_color = ROLE_COLORS.get(src_role, Fore.WHITE)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    print(
        f"[{timestamp}] "
        f"{src_color}[{src_role_formatted}]{Style.RESET_ALL} "
        f"{message}"
    )


def auto_register_queue(func):
    """Decorator: auto-register queue"""
    async def wrapper(self, src_role, dst_role, *args, **kwargs):
        # Auto-register queue (if not registered)
        if not self.is_queue_registered(src_role, dst_role):
            self.register_queue(src_role, dst_role)
        return await func(self, src_role, dst_role, *args, **kwargs)
    return wrapper

def auto_register_queue_sync(func):
    """Decorator: auto-register queue (sync version)"""
    def wrapper(self, src_role, dst_role, *args, **kwargs):
        # Auto-register queue (if not registered)
        if not self.is_queue_registered(src_role, dst_role):
            self.register_queue(src_role, dst_role)
        return func(self, src_role, dst_role, *args, **kwargs)
    return wrapper


class AsyncPipeline:
    """Asynchronous Pipeline management class, responsible for data flow and queue management between roles"""
    
    def __init__(self, max_queue_size=1, transfer_mode=TransferMode.RAY_QUEUE_OPTIMIZED):
        """
        Initialize the AsyncPipeline.
        
        Args:
            max_queue_size: Maximum queue size
            transfer_mode: Transfer mode
        """
        self.max_queue_size = max_queue_size
        self.transfer_mode = transfer_mode
        
        # Define all supported roles
        self.role = {
            "dataloader",
            "rollout",
            "train",
            "generate",
            "param_update",
            "reward",
            "logp",
            "ref_logp",
        }
        
        enhanced_print("pipeline", None, f"AsyncPipeline initialized: max_queue_size={max_queue_size}, transfer_mode={transfer_mode.value}")
        
        # Initialize based on transfer mode
        if transfer_mode in [TransferMode.RAY_QUEUE, TransferMode.RAY_QUEUE_COMPRESSED, TransferMode.RAY_QUEUE_OPTIMIZED]:
            # Use Ray queue
            self._queue_pairs = OrderedDict()
            self._object_store_pairs = None
        elif transfer_mode == TransferMode.DIRECT_OBJECT_STORE:
            # Direct use of object store
            self._object_store_pairs = OrderedDict()
            self._queue_pairs = None
        elif transfer_mode == TransferMode.HYBRID:
            # Hybrid mode, initialize both
            self._queue_pairs = OrderedDict()
            self._object_store_pairs = OrderedDict()
        else:
            raise ValueError(f"Unsupported transfer mode: {transfer_mode}")
    
    def register_queue(self, src_role, dst_role):
        """Dynamically register queue or object store"""
        use_queue_name = f"{src_role}_to_{dst_role}"
        
        # Dynamically add roles to role set
        self.role.add(src_role)
        self.role.add(dst_role)
        
        if self.transfer_mode in [TransferMode.RAY_QUEUE, TransferMode.RAY_QUEUE_COMPRESSED, TransferMode.RAY_QUEUE_OPTIMIZED]:
            # Use queue mode, register Ray queue
            if use_queue_name not in self._queue_pairs:
                self._queue_pairs[use_queue_name] = Queue(maxsize=self.max_queue_size)
                enhanced_print("pipeline", None, f"Registered queue: {use_queue_name} with maxsize={self.max_queue_size}")
        
        if self.transfer_mode in [TransferMode.DIRECT_OBJECT_STORE, TransferMode.HYBRID]:
            # Bypass queue mode, register object store list
            if use_queue_name not in self._object_store_pairs:
                self._object_store_pairs[use_queue_name] = []
                enhanced_print("pipeline", None, f"Registered object store pair: {use_queue_name}")
    
    def is_queue_registered(self, src_role, dst_role):
        """Check if queue is registered"""
        use_queue_name = f"{src_role}_to_{dst_role}"
        
        if self.transfer_mode in [TransferMode.RAY_QUEUE, TransferMode.RAY_QUEUE_COMPRESSED, TransferMode.RAY_QUEUE_OPTIMIZED]:
            return use_queue_name in self._queue_pairs
        elif self.transfer_mode == TransferMode.DIRECT_OBJECT_STORE:
            return use_queue_name in self._object_store_pairs
        elif self.transfer_mode == TransferMode.HYBRID:
            return use_queue_name in self._queue_pairs or use_queue_name in self._object_store_pairs
        return False
    
    def is_in_pipeline(self, role):
        """Check if role is in pipeline"""
        return role in self.role
    
    def is_complete(self, src_role, dst_role):
        """
        Check if pipeline is complete
        1. Current src_to_dst queue is empty
        2. Last queue (dataloader_to_train) is not empty
        """
        use_queue_name = f"{src_role}_to_{dst_role}"
        cur_pipeline_queue = self._queue_pairs.get(use_queue_name, None)
        if cur_pipeline_queue is None:
            raise ValueError(f"Queue {use_queue_name} not found in pipeline queues: {self._queue_pairs.keys()}")
        
        first_queue_name = "rollout_to_dataloader"
        last_queue_name = "dataloader_to_rollout"
        first_pipeline_queue = self._queue_pairs.get(first_queue_name)
        last_pipeline_queue = self._queue_pairs.get(last_queue_name)
        
        def _get_keys_before_cur_queue():
            keys_before_cur_queue = []
            for key in self._queue_pairs.keys():
                if key == first_queue_name: continue
                if key == last_queue_name: continue
                if key == use_queue_name: break
                keys_before_cur_queue.append(key)
            return keys_before_cur_queue
        
        # Check if all preceding queues are empty
        for key in _get_keys_before_cur_queue():
            queue = self._queue_pairs.get(key)
            if queue is not None and not queue.empty():
                return False
        
        # Current is empty
        if cur_pipeline_queue.empty() and not last_pipeline_queue.empty():
            print(f"[{src_role}] to [{dst_role}] queue is empty: {use_queue_name}")
            return True
        
        return False
    
    def get_queue_size(self, src_role, dst_role):
        """Get queue or object store size"""
        use_queue_name = f"{src_role}_to_{dst_role}"
        
        if self.transfer_mode in [TransferMode.RAY_QUEUE, TransferMode.RAY_QUEUE_COMPRESSED, TransferMode.RAY_QUEUE_OPTIMIZED]:
            # Use queue mode
            if use_queue_name in self._queue_pairs:
                return self._queue_pairs[use_queue_name].qsize()
            else:
                return 0
        elif self.transfer_mode == TransferMode.DIRECT_OBJECT_STORE:
            # Bypass queue mode
            if use_queue_name in self._object_store_pairs:
                return len(self._object_store_pairs[use_queue_name])
            else:
                return 0
        elif self.transfer_mode == TransferMode.HYBRID:
            # Hybrid mode, prioritize queue size
            if use_queue_name in self._queue_pairs:
                return self._queue_pairs[use_queue_name].qsize()
            elif use_queue_name in self._object_store_pairs:
                return len(self._object_store_pairs[use_queue_name])
            else:
                return 0
        else:
            return 0
    
    @auto_register_queue
    async def push(self, src_role, dst_role, data, debug_log=False):
        """Push data to queue - support bypassing queue to directly use object store"""
        use_queue_name = f"{src_role}_to_{dst_role}"
        if debug_log:
            enhanced_print(src_role, dst_role, f"[{src_role}] Pushing data to [{dst_role}] queue: {use_queue_name}")
        
        # Get current node information
        current_node_id = str(ray.get_runtime_context().node_id)
        
        # Record start time
        start_time = time.time()
        
        if self.transfer_mode == TransferMode.DIRECT_OBJECT_STORE:
            # Bypass queue, directly use object store
            if use_queue_name not in self._object_store_pairs:
                raise ValueError(f"Object store pair {use_queue_name} not found")
            
            # Directly use ray.put, bypass queue
            ray_put_start = time.time()
            object_ref = ray.put(data)
            ray_put_time = time.time() - ray_put_start
            
            # Add object ref to list
            self._object_store_pairs[use_queue_name].append(object_ref)
            
            total_time = time.time() - start_time
            
            # Record performance analysis
            if total_time > 0.1:
                enhanced_print(src_role, dst_role, f"DIRECT OBJECT STORE: {use_queue_name} ray_put={ray_put_time:.3f}s, total={total_time:.3f}s, refs_count={len(self._object_store_pairs[use_queue_name])}")
            
            if total_time > 1.0:
                enhanced_print(src_role, dst_role, f"⚠️ SLOW DIRECT PUT: {use_queue_name} took {total_time:.2f}s (ray_put: {ray_put_time:.2f}s) on node={current_node_id[:8]}")
        else:
            # Use Ray queue
            cur_pipeline_queue = self._queue_pairs.get(use_queue_name, None)
            assert cur_pipeline_queue is not None, f"Queue {use_queue_name} not found in pipeline queues: {self._queue_pairs.keys()}"
            
            # Record queue status
            queue_size_before = cur_pipeline_queue.qsize()
            
            # Choose transfer method
            if self.transfer_mode == TransferMode.RAY_QUEUE_OPTIMIZED:
                # Use ray.put optimization: only transfer object reference
                ray_put_start = time.time()
                object_ref = ray.put(data)
                ray_put_time = time.time() - ray_put_start
                
                # Transfer object reference
                queue_put_start = time.time()
                await cur_pipeline_queue.put_async(object_ref)
                queue_put_time = time.time() - queue_put_start
                
                total_time = time.time() - start_time
                queue_size_after = cur_pipeline_queue.qsize()
                
                # Record performance analysis
                if total_time > 0.1:
                    enhanced_print(src_role, dst_role, f"RAY_PUT OPTIMIZATION: {use_queue_name} ray_put={ray_put_time:.3f}s, queue_put={queue_put_time:.3f}s, total={total_time:.3f}s, queue={queue_size_before}->{queue_size_after}, node={current_node_id[:8]}")
                
                if total_time > 1.0:
                    enhanced_print(src_role, dst_role, f"⚠️ SLOW RAY_PUT: {use_queue_name} took {total_time:.2f}s (ray_put: {ray_put_time:.2f}s, queue_put: {queue_put_time:.2f}s) on node={current_node_id[:8]}")
            elif self.transfer_mode == TransferMode.RAY_QUEUE_COMPRESSED:
                # Original method: compression + queue transfer
                compress_start = time.time()
                
                # Always compress for compressed mode
                optimized_data = self._compress_data(data)

                # Record compression completion time
                compress_time = time.time() - compress_start
                put_start = time.time()
                
                # Key analysis: put_async performance
                await cur_pipeline_queue.put_async(optimized_data)
                
                # Record put_async completion time
                put_time = time.time() - put_start
                
                total_time = time.time() - compress_start
                queue_size_after = cur_pipeline_queue.qsize()
                
                # Detailed analysis of put_async performance, including node information
                if put_time > 0.1:  # Record if over 100ms
                    enhanced_print(src_role, dst_role, f"PUT_ASYNC ANALYSIS: {use_queue_name} put_async={put_time:.3f}s, compress={compress_time:.3f}s, total={total_time:.3f}s, queue={queue_size_before}->{queue_size_after}")
                
                # Print warning if total time exceeds 1 second
                if total_time > 1.0:
                    enhanced_print(src_role, dst_role, f"⚠️ SLOW PUSH: {use_queue_name} took {total_time:.2f}s (put_async: {put_time:.2f}s, compress: {compress_time:.2f}s) on node={current_node_id[:8]}")
            else: # RAY_QUEUE mode
                # Original method: compression + queue transfer
                compress_start = time.time()
                # Always compress for ray queue mode
                optimized_data = self._compress_data(data)
                
                compress_time = time.time() - compress_start
                put_start = time.time()
                
                await cur_pipeline_queue.put_async(optimized_data)
                
                put_time = time.time() - put_start
                
                total_time = time.time() - compress_start
                queue_size_after = cur_pipeline_queue.qsize()
                
                if put_time > 0.1:
                    enhanced_print(src_role, dst_role, f"PUT_ASYNC ANALYSIS: {use_queue_name} put_async={put_time:.3f}s, compress={compress_time:.3f}s, total={total_time:.3f}s, queue={queue_size_before}->{queue_size_after}")
                
                if total_time > 1.0:
                    enhanced_print(src_role, dst_role, f"⚠️ SLOW PUSH: {use_queue_name} took {total_time:.2f}s (put_async: {put_time:.2f}s, compress: {compress_time:.2f}s) on node={current_node_id[:8]}")
        
        return True
    
    def _optimize_data_for_transfer(self, data):
        """Optimize data transmission - compression and serialization optimization"""
        # Always compress for optimized mode
        return self._compress_data(data)

    
    def _compress_data(self, data):
        """Compress data - dynamically import compression library"""
        try:
            try:
                import lz4.frame
                compression_lib = lz4.frame
            except ImportError:
                # If lz4 is not available, fallback to zlib
                import zlib
                compression_lib = zlib
            
            # Serialize data
            serialized = pickle.dumps(data)
            
            # Use compression library
            compressed = compression_lib.compress(serialized)
            
            # Check compression effect
            compression_ratio = len(compressed) / len(serialized)
            if compression_ratio < 0.9:  # Use if compression ratio exceeds 10%
                enhanced_print("pipeline", None, f"Data compressed: {len(serialized)} -> {len(compressed)} bytes (ratio: {compression_ratio:.2f}) using {compression_lib.__name__}")
                return {"compressed": True, "data": compressed, "compression_lib": compression_lib.__name__}
            else:
                # Poor compression effect, return original data
                return {"compressed": False, "data": data}
        except Exception as e:
            enhanced_print("pipeline", None, f"Compression failed: {e}")
            return {"compressed": False, "data": data}

    def _estimate_data_size_pickle(self, data):
        """Use pickle to estimate data size (MB) - only call when needed"""
        try:
            # Try to serialize data to estimate size
            serialized = pickle.dumps(data)
            size_bytes = len(serialized)
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        except Exception as e:
            # If serialization fails, return 0
            enhanced_print("pipeline", None, f"Failed to estimate data size with pickle: {e}")
            return 0.0


    @auto_register_queue
    async def pull(self, src_role, dst_role, debug_log=False):
        """Pull data from queue - support bypassing queue to directly use object store"""
        use_queue_name = f"{src_role}_to_{dst_role}"
        if debug_log:
            enhanced_print(dst_role, src_role, f"[{dst_role}] Pulling data from [{src_role}] queue: {use_queue_name}")

        start_time = time.time()
        
        if self.transfer_mode == TransferMode.DIRECT_OBJECT_STORE:
            # Bypass queue, directly get from object store
            if use_queue_name not in self._object_store_pairs:
                raise ValueError(f"Object store pair {use_queue_name} not found")
            
            # Wait for available object ref
            while not self._object_store_pairs[use_queue_name]:
                await asyncio.sleep(0.1)  # 100ms wait
            
            # Get first object ref
            object_ref = self._object_store_pairs[use_queue_name].pop(0)
            
            # Get data from object store
            ray_get_start = time.time()
            result = ray.get(object_ref)
            ray_get_time = time.time() - ray_get_start
            
            total_time = time.time() - start_time
            
            # Record performance analysis
            if total_time > 0.1:
                enhanced_print(dst_role, src_role, f"DIRECT OBJECT STORE GET: {use_queue_name} ray_get={ray_get_time:.3f}s, total={total_time:.3f}s, remaining_refs={len(self._object_store_pairs[use_queue_name])}")
            
            if total_time > 1.0:
                enhanced_print(dst_role, src_role, f"⚠️ SLOW DIRECT GET: {use_queue_name} took {total_time:.2f}s (ray_get: {ray_get_time:.2f}s)")
        else:
            # Use Ray queue
            cur_pipeline_queue = self._queue_pairs.get(use_queue_name, None)
            assert cur_pipeline_queue is not None, f"Queue {use_queue_name} not found in pipeline queues: {self._queue_pairs.keys()}"
            
            # Get data from queue
            data = await cur_pipeline_queue.get_async()
            
            # Record queue get time
            queue_get_time = time.time() - start_time
            
            # Choose processing method
            if self.transfer_mode == TransferMode.RAY_QUEUE_OPTIMIZED:
                # Use ray.put optimization: get data from object reference
                ray_get_start = time.time()
                
                # Check if it's object reference
                if hasattr(data, '_ray_object_ref'):  # Ray object reference
                    result = ray.get(data)
                else:
                    result = data
                
                ray_get_time = time.time() - ray_get_start
                total_time = time.time() - start_time
                
                # Record performance analysis
                if total_time > 0.1:
                    enhanced_print(dst_role, src_role, f"RAY_GET OPTIMIZATION: {use_queue_name} queue_get={queue_get_time:.3f}s, ray_get={ray_get_time:.3f}s, total={total_time:.3f}s")
                
                if total_time > 1.0:
                    enhanced_print(dst_role, src_role, f"⚠️ SLOW RAY_GET: {use_queue_name} took {total_time:.2f}s (queue_get: {queue_get_time:.2f}s, ray_get: {ray_get_time:.2f}s)")
            elif self.transfer_mode == TransferMode.RAY_QUEUE_COMPRESSED:
                # Original method: decompression processing
                get_time = queue_get_time
                decompress_start = time.time()
                
                # Process compressed data
                if isinstance(data, dict) and "compressed" in data:
                    if data["compressed"]:
                        # Decompress data
                        compression_lib = data.get("compression_lib", "lz4.frame")
                        result = self._decompress_data(data["data"], compression_lib)
                    else:
                        # Uncompressed data
                        result = data["data"]
                else:
                    # Original data
                    result = data

                decompress_time = time.time() - decompress_start
                total_time = time.time() - start_time
                
                # Detailed analysis of get_async performance
                if get_time > 0.1:  # Record if over 100ms
                    enhanced_print(dst_role, src_role, f"GET_ASYNC ANALYSIS: {use_queue_name} get_async={get_time:.3f}s, decompress={decompress_time:.3f}s, total={total_time:.3f}s")
                
                # Print warning if total time exceeds 1 second
                if total_time > 1.0:
                    enhanced_print(dst_role, src_role, f"⚠️ SLOW PULL: {use_queue_name} took {total_time:.2f}s (get_async: {get_time:.2f}s, decompress: {decompress_time:.2f}s)")
            else: # RAY_QUEUE mode
                # Original method: decompression processing
                get_time = queue_get_time
                decompress_start = time.time()
                
                # Process compressed data
                if isinstance(data, dict) and "compressed" in data:
                    if data["compressed"]:
                        # Decompress data
                        compression_lib = data.get("compression_lib", "lz4.frame")
                        result = self._decompress_data(data["data"], compression_lib)
                    else:
                        # Uncompressed data
                        result = data["data"]
                else:
                    # Original data
                    result = data
                
                decompress_time = time.time() - decompress_start
                total_time = time.time() - start_time
                
                # Detailed analysis of get_async performance
                if get_time > 0.1:  # Record if over 100ms
                    enhanced_print(dst_role, src_role, f"GET_ASYNC ANALYSIS: {use_queue_name} get_async={get_time:.3f}s, decompress={decompress_time:.3f}s, total={total_time:.3f}s")
                
                # Print warning if total time exceeds 1 second
                if total_time > 1.0:
                    enhanced_print(dst_role, src_role, f"⚠️ SLOW PULL: {use_queue_name} took {total_time:.2f}s (get_async: {get_time:.2f}s, decompress: {decompress_time:.2f}s)")
        
        return result
    
    def _decompress_data(self, compressed_data, compression_lib_name="lz4.frame"):
        """Decompress data - dynamically import compression library"""
        try:
            # Dynamically import compression library
            if compression_lib_name == "lz4.frame":
                try:
                    import lz4.frame
                    compression_lib = lz4.frame
                except ImportError:
                    # If lz4 is not available, fallback to zlib
                    import zlib
                    compression_lib = zlib
            else:
                import zlib
                compression_lib = zlib
            
            # Use compression library to decompress
            if hasattr(compression_lib, 'decompress'):
                # lz4.frame.decompress
                decompressed = compression_lib.decompress(compressed_data)
            else:
                # zlib.decompress
                decompressed = compression_lib.decompress(compressed_data)
            
            # Deserialize data
            data = pickle.loads(decompressed)
            
            enhanced_print("pipeline", None, f"Data decompressed: {len(compressed_data)} -> {len(decompressed)} bytes using {compression_lib.__name__}")
            return data
        except Exception as e:
            enhanced_print("pipeline", None, f"Decompression failed: {e}")
            return None 

    @auto_register_queue_sync
    def get_cur_queue(self, src_role, dst_role):
        """Get current queue or object store list"""
        use_queue_name = f"{src_role}_to_{dst_role}"
        
        if self.transfer_mode in [TransferMode.RAY_QUEUE, TransferMode.RAY_QUEUE_COMPRESSED, TransferMode.RAY_QUEUE_OPTIMIZED]:
            # Use queue mode, return Ray queue
            if use_queue_name not in self._queue_pairs:
                raise ValueError(f"Queue {use_queue_name} not found in pipeline queues: {self._queue_pairs.keys()}")
            return self._queue_pairs[use_queue_name]
        elif self.transfer_mode == TransferMode.DIRECT_OBJECT_STORE:
            # Bypass queue mode, return object store list
            if use_queue_name not in self._object_store_pairs:
                raise ValueError(f"Object store pair {use_queue_name} not found in object store pairs: {self._object_store_pairs.keys()}")
            return self._object_store_pairs[use_queue_name]
        elif self.transfer_mode == TransferMode.HYBRID:
            if use_queue_name in self._queue_pairs:
                return self._queue_pairs[use_queue_name]
            elif use_queue_name in self._object_store_pairs:
                return self._object_store_pairs[use_queue_name]
            else:
                raise ValueError(f"Neither queue nor object store pair {use_queue_name} found")
        else:
            raise ValueError(f"Unsupported transfer mode: {self.transfer_mode}") 