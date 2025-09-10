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
Pipeline utilities and common tools for state machines.

This module contains shared utilities, decorators, and helper classes
used across different state machine implementations.
"""

import asyncio
import time
from typing import Any


class ResourceLock:
    """Resource lock, used to control resource occupancy of train/logp/ref_logp"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_owner = None
        self._waiting_queue = []
        self._train_completed_steps = set()  # Record completed train steps
        self._last_wait_log_time = {}  # Track last wait log time for each owner
        self._wait_log_interval = 10  # Log wait message every 10 seconds

    async def acquire(self, owner_name: str, step: int = None) -> bool:
        """Get resource lock"""
        if self._current_owner == owner_name:
            return True  # Already owner

        # Add to waiting queue
        if owner_name not in self._waiting_queue:
            self._waiting_queue.append(owner_name)

        async with self._lock:
            # Wait for resource available
            while self._current_owner is not None and self._current_owner != owner_name:
                current_time = time.time()
                # Only log wait message every few seconds to reduce spam
                if (
                    owner_name not in self._last_wait_log_time
                    or current_time - self._last_wait_log_time[owner_name] >= self._wait_log_interval
                ):
                    from .pipeline_utils import enhanced_print

                    enhanced_print(
                        "ResourceLock", None, f"{owner_name} waiting for resource, current owner: {self._current_owner}"
                    )
                    self._last_wait_log_time[owner_name] = current_time
                await asyncio.sleep(1)

            # Get resource
            self._current_owner = owner_name
            if owner_name in self._waiting_queue:
                self._waiting_queue.remove(owner_name)
            # Clear wait log time when acquired
            if owner_name in self._last_wait_log_time:
                del self._last_wait_log_time[owner_name]
            from .pipeline_utils import enhanced_print

            enhanced_print("ResourceLock", None, f"{owner_name} acquired resource lock")
            return True

    async def release(self, owner_name: str, step: int = None):
        """Release resource lock"""
        if self._current_owner == owner_name:
            self._current_owner = None

            # If train completed, record step
            if owner_name == "train" and step is not None:
                self._train_completed_steps.add(step)
                from .pipeline_utils import enhanced_print

                enhanced_print("ResourceLock", None, f"Train step {step} completed")

            from .pipeline_utils import enhanced_print

            enhanced_print("ResourceLock", None, f"{owner_name} released resource lock")
            if self._waiting_queue:
                enhanced_print("ResourceLock", None, f"Next in queue: {self._waiting_queue[0]}")
        else:
            from .pipeline_utils import enhanced_print

            enhanced_print(
                "ResourceLock", None, f"Warning: {owner_name} tried to release lock owned by {self._current_owner}"
            )

    def get_status(self) -> dict[str, Any]:
        """Get lock status"""
        return {
            "current_owner": self._current_owner,
            "waiting_queue": self._waiting_queue.copy(),
            "train_completed_steps": sorted(self._train_completed_steps),
        }


class TimingStatsCollector:
    """Tool class to collect and manage timing statistics data"""

    def __init__(self):
        self.stats = {}
        self.step_count = 0

    def record_timing(self, role_name: str, step: Any, duration: float):
        """Record execution time"""
        if role_name not in self.stats:
            self.stats[role_name] = {
                "total_time": 0.0,
                "count": 0,
                "avg_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "recent_times": [],
            }

        stats = self.stats[role_name]
        stats["total_time"] += duration
        stats["count"] += 1
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)

        # Keep recent 10 execution times
        stats["recent_times"].append(duration)
        if len(stats["recent_times"]) > 10:
            stats["recent_times"].pop(0)

        # Print real-time statistics, but reduce frequency
        if stats["count"] % 10 == 0 or duration > 1.0:  # Print every 10 times or more than 1 second
            from .pipeline_utils import enhanced_print

            enhanced_print(
                role_name,
                None,
                f"Step {step}: process_data took {duration:.2f}s "
                f"(avg: {stats['avg_time']:.2f}s, min: {stats['min_time']:.2f}s, max: {stats['max_time']:.2f}s)",
            )

    def get_summary(self) -> dict[str, Any]:
        """Get statistics summary"""
        summary = {}
        for role_name, stats in self.stats.items():
            summary[role_name] = {
                "avg_time": stats["avg_time"],
                "min_time": stats["min_time"],
                "max_time": stats["max_time"],
                "total_count": stats["count"],
                "recent_avg": sum(stats["recent_times"]) / len(stats["recent_times"]) if stats["recent_times"] else 0.0,
            }
        return summary

    def print_summary(self):
        """Print statistics summary"""
        print("\n" + "=" * 80)
        print("PROCESS_DATA TIMING SUMMARY")
        print("=" * 80)
        for role_name, stats in self.stats.items():
            recent_avg = sum(stats["recent_times"]) / len(stats["recent_times"]) if stats["recent_times"] else 0.0
            print(
                f"{role_name:15} | "
                f"Count: {stats['count']:4d} | "
                f"Avg: {stats['avg_time']:6.3f}s | "
                f"Min: {stats['min_time']:6.3f}s | "
                f"Max: {stats['max_time']:6.3f}s | "
                f"Recent: {recent_avg:6.3f}s"
            )
        print("=" * 80)


def timing_decorator(role_name: str):
    """Decorator: add timing statistics to process_data method"""

    def decorator(func):
        async def wrapper(self, data: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(self, data)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time

                # Extract step information for logging
                step_info = "N/A"
                if hasattr(self, "trainer"):
                    if role_name == "dataloader":
                        step_info = self.trainer.dataloader_global_step
                    elif role_name == "generate":
                        step_info = self.trainer.generate_global_step
                    else:
                        step_info = self.trainer.global_steps
                elif isinstance(data, tuple) and len(data) > 0:
                    step_info = data[0]
                elif isinstance(data, dict) and "step" in data:
                    step_info = data["step"]

                # Use global timing collector
                if hasattr(self, "timing_collector"):
                    self.timing_collector.record_timing(role_name, step_info, duration)
                else:
                    # If no timing_collector, use global instance
                    global_timing_collector.record_timing(role_name, step_info, duration)

        return wrapper

    return decorator


def is_ref_model_separated(sperated_node_tasks):
    """
    Determine if ref model is separated based on sperated_node_tasks configuration.

    Args:
        sperated_node_tasks: omegaconf.listconfig.ListConfig or list, task separation configuration

    Returns:
        bool: True if ref model is separated, False otherwise
    """
    # Enhanced check for separated ref model based on sperated_node_tasks configuration
    # sperated_node_tasks is omegaconf.listconfig.ListConfig type

    # Check if ref_logp is in a separate task group
    if hasattr(sperated_node_tasks, "__iter__"):
        # Handle ListConfig type
        sperated_task_list = list(sperated_node_tasks)
        if len(sperated_task_list) > 0:
            if isinstance(sperated_task_list[0], list):
                # Nested list case: [[logp, actor-train], ref_logp, generate]
                if "ref_logp" in sperated_task_list[0]:
                    # ref_logp is in the first group with actor-train, not separated
                    return False
                else:
                    # Check if ref_logp is in any other group
                    for task_group in sperated_task_list:
                        if isinstance(task_group, list) and "ref_logp" in task_group:
                            # ref_logp is in a group with other tasks, not separated
                            return False
                    # ref_logp is not in any group, it's separated
                    return True
            else:
                # Flat list case: [logp, ref_logp, actor-train, generate]
                if "ref_logp" in sperated_task_list:
                    # ref_logp is explicitly listed, it's separated
                    return True
                else:
                    # ref_logp is not explicitly listed, not separated
                    return False
        else:
            # Empty list case, not separated
            return False
    else:
        # Fallback for non-iterable types, not separated
        return False


def calculate_pool_sizes_from_task_groups(sperated_node_tasks, sperated_node_ratios, total_ngpus):
    """
    Calculate pool sizes based on task groups and node ratios.

    Args:
        sperated_node_tasks: omegaconf.listconfig.ListConfig or list, task separation configuration
        sperated_node_ratios: list, node ratios for each task group
        total_ngpus: int, total number of GPUs

    Returns:
        dict: pool sizes for each task group
    """
    if not hasattr(sperated_node_tasks, "__iter__"):
        # Fallback: use original logic
        return {
            "logp": int(sperated_node_ratios[0] * total_ngpus),
            "actor-train": int(sperated_node_ratios[1] * total_ngpus),
            "ref_logp": int(sperated_node_ratios[2] * total_ngpus),
            "generate": int(sperated_node_ratios[3] * total_ngpus),
        }

    sperated_task_list = list(sperated_node_tasks)

    # Assert that the number of task groups matches the number of node ratios
    assert len(sperated_task_list) == len(sperated_node_ratios), (
        f"Number of task groups ({len(sperated_task_list)}) must match number of node ratios "
        f"({len(sperated_node_ratios)})"
    )

    pool_sizes = {}

    # Calculate pool sizes based on task groups
    for i, task_group in enumerate(sperated_task_list):
        ratio = sperated_node_ratios[i]
        pool_size = int(ratio * total_ngpus)

        # Ensure pool_size is at least 1
        if pool_size <= 0:
            pool_size = 1
            print(
                f"Warning: Pool size for task group {task_group} was {int(ratio * total_ngpus)}, "
                f"setting to minimum value 1"
            )

        # Check if task_group is iterable (list, ListConfig, etc.)
        if hasattr(task_group, "__iter__") and not isinstance(task_group, str):
            # Nested list case: assign pool size to each task in the group
            for task in task_group:
                pool_sizes[task] = pool_size
        else:
            # Single task case
            pool_sizes[task_group] = pool_size

    return pool_sizes


# Global instance
resource_lock = ResourceLock()  # For train/logp/ref_logp/param_update
engine_resource_lock = ResourceLock()  # For generate/validation
dataloader_scheduler_lock = ResourceLock()  # For dataloader scheduling (train vs validation)
global_timing_collector = TimingStatsCollector()
