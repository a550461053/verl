"""
Asynchronous communication manager based on Ray put/get
Coordinate with multi-threading to achieve overlap, avoid distributed communication deadlock
"""

import time

import ray


def cross_process_ray_put(data, version: int = None):
    """Cross-process Ray put operation - return object_ref for other processes to use"""

    # Record start time for performance profiling
    put_start = time.time()
    object_ref = ray.put([data])  # Ray doesn't support direct put of ref-object, wrap data in list
    put_time = time.time() - put_start

    return object_ref


def cross_process_ray_get(object_ref):
    """Cross-process Ray get operation - directly use object_ref to get data, and clean up object ref"""
    # Record start time for performance profiling
    get_ref_start = time.time()
    object = ray.get(object_ref)[0]
    get_ref_time = time.time() - get_ref_start

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
