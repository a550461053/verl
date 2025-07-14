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
# limitations under the License.

"""
Asynchronous communication manager based on Ray put/get
Coordinate with multi-threading to achieve overlap, avoid distributed communication deadlock
"""

import threading
import time
import torch
import ray
from typing import Dict, List, Optional, Any, Union
from enum import Enum, auto
import queue
import pickle


def cross_process_ray_put(data, version: int = None):
    """Cross-process Ray put operation - return object_ref for other processes to use"""
    import ray
    import time
    
    # Record start time for performance profiling    
    put_start = time.time()
    object_ref = ray.put([data])  # Ray doesn't support direct put of ref-object, wrap data in list
    put_time = time.time() - put_start
    
    # print(f"[CrossProcessRayPut] Version {version}: put={put_time*1000:.2f}ms")
    
    return object_ref

def cross_process_ray_get(object_ref):
    """Cross-process Ray get operation - directly use object_ref to get data, and clean up object ref"""
    import ray
    import time
    
    # Record start time for performance profiling
    get_ref_start = time.time()
    object = ray.get(object_ref)[0]
    get_ref_time = time.time() - get_ref_start
    
    # After getting data, clean up object ref reference, let Ray auto garbage collection
    # Note: Ray's object refs are auto garbage collected, we just need to ensure we don't hold references
    # del object_ref
    
    # print(f"[CrossProcessRayGet] get_ref={get_ref_time*1000:.2f}ms")
    return object

def create_store_refs_queue():
    """Create store refs queue"""
    import ray
    queue = ray.util.queue.Queue(maxsize=10)
    # print(f"[StoreRefsQueue] Created store refs queue")
    return queue

def put_store_refs_to_queue(queue, store_refs):
    """Put store refs into queue"""
    try:
        # Clear queue, only keep latest store refs
        # while not queue.empty():
        #     queue.get_nowait()
        
        # Put new store refs
        queue.put(store_refs)
        # print(f"[StoreRefsQueue] Put {len(store_refs)} store refs to queue")
        return True
    except Exception as e:
        print(f"[StoreRefsQueue] Failed to put store refs: {e}")
        return False

def get_store_refs_from_queue(queue):
    """Get store refs from queue"""
    try:
        # Blocking get, wait for store refs to be available
        store_refs = queue.get()
        # print(f"[StoreRefsQueue] Got {len(store_refs)} store refs from queue")
        return store_refs
    except Exception as e:
        print(f"[StoreRefsQueue] Failed to get store refs: {e}")
        return None

 