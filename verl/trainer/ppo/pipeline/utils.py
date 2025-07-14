"""
Pipeline utilities and common tools for state machines.

This module contains shared utilities, decorators, and helper classes
used across different state machine implementations.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from enum import Enum, auto


class ResourceLock:
    """Resource lock, used to control resource occupancy of train/logp/ref_logp"""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_owner = None
        self._waiting_queue = []
        self._train_completed_steps = set()  # Record completed train steps
    
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
                from .pipeline_utils import enhanced_print
                enhanced_print("ResourceLock", None, f"{owner_name} waiting for resource, current owner: {self._current_owner}")
                await asyncio.sleep(1)
            
            # Get resource
            self._current_owner = owner_name
            if owner_name in self._waiting_queue:
                self._waiting_queue.remove(owner_name)
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
                enhanced_print("ResourceLock", None, f"Train step {step} completed, available steps: {sorted(self._train_completed_steps)}")
            
            from .pipeline_utils import enhanced_print
            enhanced_print("ResourceLock", None, f"{owner_name} released resource lock")
            if self._waiting_queue:
                enhanced_print("ResourceLock", None, f"Next in queue: {self._waiting_queue[0]}")
        else:
            from .pipeline_utils import enhanced_print
            enhanced_print("ResourceLock", None, f"Warning: {owner_name} tried to release lock owned by {self._current_owner}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get lock status"""
        return {
            "current_owner": self._current_owner,
            "waiting_queue": self._waiting_queue.copy(),
            "train_completed_steps": sorted(self._train_completed_steps)
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
                'total_time': 0.0,
                'count': 0,
                'avg_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'recent_times': []
            }
        
        stats = self.stats[role_name]
        stats['total_time'] += duration
        stats['count'] += 1
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)
        
        # Keep recent 10 execution times
        stats['recent_times'].append(duration)
        if len(stats['recent_times']) > 10:
            stats['recent_times'].pop(0)
        
        # Print real-time statistics, but reduce frequency
        if stats['count'] % 10 == 0 or duration > 1.0:  # Print every 10 times or more than 1 second
            from .pipeline_utils import enhanced_print
            enhanced_print(role_name, None, f"Step {step}: process_data took {duration:.2f}s "
                  f"(avg: {stats['avg_time']:.2f}s, min: {stats['min_time']:.2f}s, max: {stats['max_time']:.2f}s)")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        summary = {}
        for role_name, stats in self.stats.items():
            summary[role_name] = {
                'avg_time': stats['avg_time'],
                'min_time': stats['min_time'],
                'max_time': stats['max_time'],
                'total_count': stats['count'],
                'recent_avg': sum(stats['recent_times']) / len(stats['recent_times']) if stats['recent_times'] else 0.0
            }
        return summary
    
    def print_summary(self):
        """Print statistics summary"""
        print("\n" + "="*80)
        print("PROCESS_DATA TIMING SUMMARY")
        print("="*80)
        for role_name, stats in self.stats.items():
            recent_avg = sum(stats['recent_times']) / len(stats['recent_times']) if stats['recent_times'] else 0.0
            print(f"{role_name:15} | "
                  f"Count: {stats['count']:4d} | "
                  f"Avg: {stats['avg_time']:6.3f}s | "
                  f"Min: {stats['min_time']:6.3f}s | "
                  f"Max: {stats['max_time']:6.3f}s | "
                  f"Recent: {recent_avg:6.3f}s")
        print("="*80)


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
                if hasattr(self, 'trainer'):
                    if role_name == "dataloader":
                        step_info = self.trainer.dataloader_global_step
                    elif role_name == "generate":
                        step_info = self.trainer.generate_global_step
                    else:
                        # global step has been updated in the trainer
                        step_info = self.trainer.global_steps - 1
                elif isinstance(data, tuple) and len(data) > 0:
                    step_info = data[0]
                elif isinstance(data, dict) and 'step' in data:
                    step_info = data['step']
                
                # Use global timing collector
                if hasattr(self, 'timing_collector'):
                    self.timing_collector.record_timing(role_name, step_info, duration)
                else:
                    # If no timing_collector, use global instance
                    global_timing_collector.record_timing(role_name, step_info, duration)
        
        return wrapper
    return decorator


# Global instance
resource_lock = ResourceLock()
global_timing_collector = TimingStatsCollector() 