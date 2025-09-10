"""
Pipeline module for PPO training with async pipeline support.
"""

from .pipeline_utils import (
    PIPELINE_END_SIGNAL,
    PIPELINE_START_SINGLE,
    ROLE_COLORS,
    AsyncPipeline,
    enhanced_print,
)
from .state_machine import (
    AsyncTrainingFlow,
    BaseRoleStateMachine,
    RoleEvent,
    RoleState,
)
from .state_machine_impl import (
    DataloaderStateMachine,
    GenerateStateMachine,
    LogPStateMachine,
    ParamUpdateStateMachine,
    RefLogPStateMachine,
    RewardStateMachine,
    RolloutStateMachine,
    TrainStateMachine,
    create_role_state_machine,
)
from .utils import (
    ResourceLock,
    TimingStatsCollector,
    global_timing_collector,
    resource_lock,
    timing_decorator,
)

__all__ = [
    # Pipeline utility classes
    "AsyncPipeline",
    "enhanced_print",
    "PIPELINE_END_SIGNAL",
    "PIPELINE_START_SINGLE",
    "ROLE_COLORS",
    # State Machine basic framework
    "RoleState",
    "RoleEvent",
    "BaseRoleStateMachine",
    "AsyncTrainingFlow",
    # State Machine implementation
    "DataloaderStateMachine",
    "TrainStateMachine",
    "RolloutStateMachine",
    "RewardStateMachine",
    "LogPStateMachine",
    "RefLogPStateMachine",
    "ParamUpdateStateMachine",
    "GenerateStateMachine",
    "create_role_state_machine",
    # Utility classes
    "ResourceLock",
    "TimingStatsCollector",
    "timing_decorator",
    "resource_lock",
    "global_timing_collector",
]
