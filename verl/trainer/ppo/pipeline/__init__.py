"""
Pipeline module for PPO training with async pipeline support.
"""

from .pipeline_utils import (
    AsyncPipeline,
    enhanced_print,
    PIPELINE_END_SIGNAL,
    PIPELINE_START_SINGLE,
    ROLE_COLORS,
)

from .state_machine import (
    RoleState,
    RoleEvent,
    BaseRoleStateMachine,
    AsyncTrainingFlow,
)

from .state_machine_impl import (
    DataloaderStateMachine,
    TrainStateMachine,
    RolloutStateMachine,
    RewardStateMachine,
    LogPStateMachine,
    RefLogPStateMachine,
    ParamUpdateStateMachine,
    GenerateStateMachine,
    create_role_state_machine,
)

from .utils import (
    ResourceLock,
    TimingStatsCollector,
    timing_decorator,
    resource_lock,
    global_timing_collector,
)

__all__ = [
    # Pipeline 工具类
    "AsyncPipeline",
    "enhanced_print", 
    "PIPELINE_END_SIGNAL",
    "PIPELINE_START_SINGLE",
    "ROLE_COLORS",
    
    # State Machine 基础框架
    "RoleState",
    "RoleEvent", 
    "BaseRoleStateMachine",
    "AsyncTrainingFlow",
    
    # State Machine 实现
    "DataloaderStateMachine",
    "TrainStateMachine",
    "RolloutStateMachine",
    "RewardStateMachine",
    "LogPStateMachine",
    "RefLogPStateMachine",
    "ParamUpdateStateMachine",
    "GenerateStateMachine",
    "create_role_state_machine",
    
    # 工具类
    "ResourceLock",
    "TimingStatsCollector",
    "timing_decorator",
    "resource_lock",
    "global_timing_collector",
] 