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
"""
Asynchronous communication manager based on Ray put/get
Coordinate with multi-threading to achieve overlap, avoid distributed communication deadlock
"""

import ray


def cross_process_ray_put(data, version: int = None):
    """Cross-process Ray put operation - return object_ref for other processes to use"""

    # Record start time for performance profiling
    object_ref = ray.put([data])  # Ray doesn't support direct put of ref-object, wrap data in list

    return object_ref


def cross_process_ray_get(object_ref):
    """Cross-process Ray get operation - directly use object_ref to get data, and clean up object ref"""
    # Record start time for performance profiling
    object = ray.get(object_ref)[0]

    return object


def create_store_refs_queue():
    """Create store refs queue"""
    queue = ray.util.queue.Queue(maxsize=10)
    return queue


def put_store_refs_to_queue(queue, store_refs):
    """Put store refs into queue"""
    try:
        # Clear queue, only keep latest store refs
        # while not queue.empty():
        #     queue.get_nowait()

        # Put new store refs
        queue.put(store_refs)
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
