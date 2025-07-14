"""
State machine implementation for async pipeline roles.

This module provides the base state machine framework and common utilities.
The actual state machine implementations are in enhanced_state_machine.py.
"""

import asyncio
import time
import ray
from enum import Enum, auto
from typing import Optional, Any, Dict, List
from abc import ABC, abstractmethod

from .pipeline_utils import enhanced_print

"""
State Machine Design:

IDLE
 └── START ──> WAITING_INPUT
                  ├── RECEIVE_DATA ──> PROCESSING
                  │                      ├── PROCESS_COMPLETE ──> WAITING_OUTPUT
                  │                      └── ERROR ──> ERROR
                  ├── STOP ──> DONE
                  └── ERROR ──> ERROR
WAITING_OUTPUT
 ├── SEND_DATA ──> WAITING_INPUT
 └── ERROR ──> ERROR
ERROR
 └── START ──> WAITING_INPUT
DONE
 └── START ──> WAITING_INPUT
"""


class RoleState(Enum):
    """Role state enumeration"""
    IDLE = auto()           # Idle state
    WAITING_INPUT = auto()  # Waiting for input
    PROCESSING = auto()     # Processing
    WAITING_OUTPUT = auto() # Waiting for output
    DONE = auto()          # Done
    ERROR = auto()         # Error state


class RoleEvent(Enum):
    """Role event enumeration"""
    START = auto()         # Start
    RECEIVE_DATA = auto()  # Received data
    PROCESS_COMPLETE = auto()  # Process complete
    SEND_DATA = auto()     # Send data
    ERROR = auto()         # Error
    STOP = auto()          # Stop


class BaseRoleStateMachine(ABC):
    """Base role state machine class"""
    
    def __init__(self, role_name: str, pipeline):
        """
        Initialize state machine
        
        Args:
            role_name: Role name
            pipeline: AsyncPipeline instance
        """
        self.role_name = role_name
        self.pipeline = pipeline
        self.state = RoleState.IDLE
        self.current_data = None
        self.error_message = None
        self.metrics = {}
        
        # State transition mapping
        self.transitions = {
            RoleState.IDLE: {
                RoleEvent.START: RoleState.WAITING_INPUT,
            },
            RoleState.WAITING_INPUT: {
                RoleEvent.RECEIVE_DATA: RoleState.PROCESSING,
                RoleEvent.STOP: RoleState.DONE,
            },
            RoleState.PROCESSING: {
                RoleEvent.PROCESS_COMPLETE: RoleState.WAITING_OUTPUT,
                RoleEvent.ERROR: RoleState.ERROR,
            },
            RoleState.WAITING_OUTPUT: {
                RoleEvent.SEND_DATA: RoleState.WAITING_INPUT,
                RoleEvent.ERROR: RoleState.ERROR,
            },
            RoleState.ERROR: {
                RoleEvent.START: RoleState.WAITING_INPUT,
            },
            RoleState.DONE: {
                RoleEvent.START: RoleState.WAITING_INPUT,
            },
        }
    
    def can_transition(self, event: RoleEvent) -> bool:
        """Check if state transition can be executed"""
        return event in self.transitions.get(self.state, {})
    
    def transition(self, event: RoleEvent) -> bool:
        """Execute state transition"""
        if self.can_transition(event):
            old_state = self.state
            self.state = self.transitions[self.state][event]
            self._on_state_change(old_state, self.state, event)
            return True
        return False
    
    def _on_state_change(self, old_state: RoleState, new_state: RoleState, event: RoleEvent):
        """State change callback"""
        enhanced_print(self.role_name, self.role_name, f"Event: {event.name} (State: {old_state.name} -> {new_state.name})")
    
    @abstractmethod
    async def process_data(self, data: Any) -> Any:
        """Abstract method to process data, subclass must implement"""
        pass
    
    @abstractmethod
    async def get_input_data(self) -> Optional[Any]:
        """Abstract method to get input data, subclass must implement"""
        pass
    
    @abstractmethod
    async def send_output_data(self, data: Any) -> bool:
        """Abstract method to send output data, subclass must implement"""
        pass
    
    async def step(self) -> bool:
        """Execute a state machine step"""
        if self.state == RoleState.IDLE:
            if self.transition(RoleEvent.START):
                return True
                
        elif self.state == RoleState.WAITING_INPUT:
            data = await self.get_input_data()
            if data is not None:
                self.current_data = data
                if self.transition(RoleEvent.RECEIVE_DATA):
                    return True
                    
        elif self.state == RoleState.PROCESSING:
            result = await self.process_data(self.current_data)
            if result is not None:
                self.current_data = result
                if self.transition(RoleEvent.PROCESS_COMPLETE):
                    return True
                    
        elif self.state == RoleState.WAITING_OUTPUT:
            success = await self.send_output_data(self.current_data)
            if success:
                self.current_data = None
                if self.transition(RoleEvent.SEND_DATA):
                    return True
                    
        elif self.state == RoleState.ERROR:
            # Error state - terminate directly, no retry
            enhanced_print(self.role_name, None, f"Error state reached, terminating {self.role_name} state machine")
            self.state = RoleState.DONE
            return False
            
        elif self.state == RoleState.DONE:
            return False  # End loop
            
        return True
    
    async def run(self):
        """Run state machine main loop"""
        enhanced_print(self.role_name, None, f"Starting {self.role_name} state machine")
        
        while self.state != RoleState.DONE:
            await self.step()
        
        enhanced_print(self.role_name, None, f"Stopped {self.role_name} state machine")
    
    async def stop(self):
        """Stop state machine"""
        enhanced_print(self.role_name, None, f"Stopping {self.role_name} state machine")
        self.state = RoleState.DONE


# ============================================================================
# Training flow management class
# ============================================================================

from .pipeline_utils import AsyncPipeline, PIPELINE_END_SIGNAL, PIPELINE_START_SINGLE, TransferMode
from .utils import global_timing_collector


class AsyncTrainingFlow:
    """Training flow example, support blocking mode, async param sync, fully async mode and async RL optimization"""
    
    def __init__(self, trainer=None, use_async_rl=True):
        self.trainer = trainer
        self.use_async_rl = use_async_rl
        
        # Select pipeline type based on switch
        if use_async_rl:
            # Async RL mode uses direct object store communication for maximum performance
            self.pipeline = AsyncPipeline(
                max_queue_size=5,  # Significantly increase queue size
                transfer_mode=TransferMode.DIRECT_OBJECT_STORE  # Directly use object store
            )
            enhanced_print("AsyncTrainingFlow", None, 
                         "Using async RL mode pipeline with direct object store communication for maximum performance")
        else:
            self.pipeline = AsyncPipeline(
                max_queue_size=5,  # Increase queue size
                transfer_mode=TransferMode.RAY_QUEUE_COMPRESSED  # Compression mode
            )
            enhanced_print("AsyncTrainingFlow", None, 
                         "Using async mode pipeline with compression")
        
        # Create state machines
        self.state_machines = self._create_state_machines()
        
        # Performance statistics collector
        self.timing_collector = global_timing_collector
        
        # Print resource lock configuration
        enhanced_print("AsyncTrainingFlow", None, 
                     "Using resource lock mechanism for train/logp/ref_logp (shared cluster resources)")
    
    def _create_state_machines(self):
        """Create state machine instance"""
        if self.use_async_rl:
            enhanced_print("AsyncTrainingFlow", None, 
                         "Using async RL state machines for dual buffer and async param sync")
        else:
            enhanced_print("AsyncTrainingFlow", None, 
                         "Using async state machines (legacy mode - not recommended)")
        
        # Import state machine implementation
        from .state_machine_impl import create_role_state_machine
        
        return {
            role_name: create_role_state_machine(
                role_name, self.pipeline, self.trainer, 
                use_async_rl=self.use_async_rl
            )
            for role_name in ["dataloader", "rollout", "train", "generate", "reward", "logp", "ref_logp", "param_update"]
        }
    
    async def run_state_machine_pipeline(self):
        """Run state machine pipeline, support all modes"""
        print("\n" + "="*60)
        print("STATE MACHINE PIPELINE TRAINING FLOW")
        print("="*60)
        
        pipeline_type = "Efficient"
        
        if self.use_async_rl:
            mode_type = "Async RL"
            features = "🚀 Async RL, dual buffer, async param sync, 1.5-2x performance"
        else:
            mode_type = "Async (Legacy)"
            features = "⚠️  Async mode (legacy), may have timing issues"
        
        print(f"Pipeline Type: {pipeline_type}")
        print(f"Mode: {mode_type}")
        print(f"State Machines: {len(self.state_machines)}")
        print(f"Features: {features}")
        
        # Print all state machine information
        for role_name, sm in self.state_machines.items():
            print(f"  - {role_name}: {type(sm).__name__}")
        
        await self._init_before_pipeline()
        
        try:
            # Run all state machines
            state_machine_tasks = [sm.run() for sm in self.state_machines.values()]
            await asyncio.gather(*state_machine_tasks)
            
            print("State machine pipeline training completed!")
            
            # Output timing statistics summary
            global_timing_collector.print_summary()
            
            # Output param_update specific statistics
            param_update_sm = self.state_machines.get("param_update")
            if hasattr(param_update_sm, 'get_status_info'):
                status = param_update_sm.get_status_info()
                print(f"\nParam Update Stats: {status}")
                
        except Exception as e:
            print(f"Error in state machine pipeline: {e}")
            import traceback
            traceback.print_exc()
            
            # Try graceful shutdown
            print("Attempting graceful shutdown...")
            try:
                # Send end signal to all state machines
                for role_name, sm in self.state_machines.items():
                    try:
                        if hasattr(sm, 'stop'):
                            await sm.stop()
                    except Exception as stop_error:
                        print(f"Error stopping {role_name} state machine: {stop_error}")
                
                # Output timing statistics summary
                global_timing_collector.print_summary()
                
            except Exception as shutdown_error:
                print(f"Error during graceful shutdown: {shutdown_error}")
            
            # Re-raise exception
            raise e
    
    async def _init_before_pipeline(self):
        # trainer global step start from 1
        self.trainer.dataloader_global_step = 0
        self.trainer.generate_global_step = 0

        # Start signal, fully align with ray_async_pipeline_trainer.py
        await self.pipeline.push("rollout", "dataloader", PIPELINE_START_SINGLE)
        
        # First param_update, ensure generate waits
        enhanced_print("AsyncTrainingFlow", None, "Starting first param_update to ensure generate waits...")
        await self.pipeline.push(src_role="train", dst_role="param_update", data=self.trainer.global_steps)

        # If logp/ref_logp and train share resources, block push to logp/ref_logp, ensure no resource contention
        if self.trainer.config.trainer.get("share_resource_between_train_logp_ref_logp", True):
            await self.pipeline.push(src_role="train", dst_role="logp", data=self.trainer.global_steps)
            await self.pipeline.push(src_role="train", dst_role="ref_logp", data=self.trainer.global_steps)
            enhanced_print("train", None, f"Sent training completion signal for step {self.trainer.global_steps} to logp/ref_logp")
        

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        pipeline_type = "efficient"
        
        if self.use_async_rl:
            mode = "async_rl"
        else:
            mode = "blocking"
        
        status = {
            "pipeline_type": pipeline_type,
            "mode": mode,
            "use_async_rl": self.use_async_rl,
            "state_machines": len(self.state_machines),
            "trainer_steps": self.trainer.global_steps,
            "trainer_dataloader_steps": self.trainer.dataloader_global_step,
            "trainer_generate_steps": self.trainer.generate_global_step,
            "state_machine_states": {},
            "timing_stats": global_timing_collector.get_summary()
        }
        
        for role_name, sm in self.state_machines.items():
            status["state_machine_states"][role_name] = sm.state.value
            
        # Add param_update specific status
        param_update_sm = self.state_machines.get("param_update")
        if hasattr(param_update_sm, 'get_status_info'):
            status["param_update_details"] = param_update_sm.get_status_info()
        
        return status
