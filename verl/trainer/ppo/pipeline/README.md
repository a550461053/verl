# Async-RL Pipeline

## Overview

The Async-RL Pipeline is a state-of-the-art implementation of asynchronous reinforcement learning training based on a fully decoupled architecture. It separates actor-train, actor-forward-logp, ref_logp, and rollout-generate components to achieve optimal performance and scalability.

## Architecture Principles

### Fully Decoupled Architecture
The system is built on a fully decoupled RL training architecture where different components operate independently:
- **Actor-Train**: Handles the main training loop
- **Actor-Forward-LogP**: Computes log probabilities for the actor
- **Ref_LogP**: Computes reference log probabilities
- **Rollout-Generate**: Handles sequence generation and rollout

This decoupling enables true asynchronous operation and eliminates bottlenecks that occur in traditional synchronous RL training.

## Key Features

### 1. State Machine Design

The pipeline implements a sophisticated state machine design where different state transitions correspond to the entire async-RL pipeline workflow:

**Pipeline Components:**
- `dataloader` → `generate` → `rollout` → `logp` → `ref_logp` → `reward` → `train` → `param_update`

**Design Rationale:**
RL training workflows are inherently complex. While synchronous approaches can simply execute tasks sequentially, async-RL requires complex state transitions between different tasks. To ensure both performance and accuracy, the system employs flexible scheduling strategies that bind tasks to resources logically. Each task maintains its own production and consumption loop to prevent errors. In this context, designing RL state machines provides a friendly and manageable approach.

**Benefits:**
- Clear separation of concerns
- Predictable state transitions
- Error isolation and recovery
- Resource management optimization

### 2. Asynchronous Parameter Synchronization

The system implements true asynchronous parameter updates by decomposing the parameter synchronization process:

**Traditional Approach:**
- Used NCCL-based parameter synchronization
- Limited by NCCL's non-thread-safe nature
- Could not achieve true asynchrony

**New Implementation:**
The parameter update process is decomposed into three main components:
1. **Gather**: Uses NCCL for parameter aggregation (must be serial)
2. **Send/Recv**: Asynchronous CPU communication
3. **Load**: Parameter loading without affecting GPU compute

**Benefits:**
- Enables `generate` vs `param_update` vs `train` asynchrony
- Preserves GPU compute resources
- Maintains training accuracy
- Reuses existing VERL implementation logic

### 3. Arbitrary Granularity Parallelism

The system addresses RL training bottlenecks through intelligent task overlap:

**Problem:**
RL bottlenecks typically occur in `rollout` or `train` tasks. When any task blocks (e.g., long-tail issues causing generate tasks to pend), all other GPUs idle, significantly reducing training efficiency.

**Solution:**
- Complete asynchrony decouples train and rollout tasks
- Allows off-policy training with task overlap
- Enables optimal performance through intelligent scheduling

**Example Scenarios:**

**Fast Generation Tasks:**
- Rollout-generate completes quickly while train is still running
- System can proceed to next round or even n rounds of generate tasks
- Maintains continuous training flow

**Slow Generation Tasks:**
- Long generate tasks don't block train operations
- Train can continue consuming previous rounds' generate results
- Ensures sustained training progress

**Performance Impact:**
- Reduces long-tail effects
- Maintains high Model FLOPs Utilization (MFU)
- Achieves near-linear scaling (0.9 linearity)

## Performance Results

### Benchmark Configuration
- **Model**: Red-MoE-16B
- **Hardware**: 4 machines
- **Configuration**: TP1 + PP1 + EP4 + SGLang-TP2
- **Algorithm**: GRPO
- **Batch Size**: 128

### Performance Improvements
- **Baseline**: Synchronous RL training
- **Improvement**: 50% performance increase
- **Scalability**: Increasing batch size can achieve up to 100% performance improvement

## Pipeline Components

### State Machines

1. **DataloaderStateMachine**
   - Manages data loading and preprocessing
   - Ensures proper data flow to downstream components

2. **GenerateStateMachine**
   - Handles sequence generation
   - Supports interruptible generation for better resource utilization

3. **RolloutStateMachine**
   - Orchestrates the rollout process
   - Manages data flow between components

4. **LogPStateMachine**
   - Computes log probabilities for the actor
   - Ensures proper resource locking

5. **RefLogPStateMachine**
   - Computes reference log probabilities
   - Supports reference policy evaluation

6. **RewardStateMachine**
   - Calculates rewards and advantages
   - Handles reward model integration

7. **TrainStateMachine**
   - Manages the main training loop
   - Coordinates all training operations

8. **ParamUpdateStateMachine**
   - Handles asynchronous parameter updates
   - Manages parameter synchronization

### Pipeline Flow

```
dataloader → generate → rollout → logp/ref_logp → reward → train → param_update
    ↓           ↓         ↓           ↓           ↓        ↓         ↓
  Data      Sequence   Process    Compute    Calculate  Update   Sync
Loading   Generation   Rollout   Log Probs   Rewards   Model    Params
```

## Configuration

### Key Parameters

```python
# Async RL Configuration
use_async_rl: bool = True
enable_param_sync: bool = True
enable_async_mode: bool = True

# Resource Management
share_resource_between_train_logp_ref_logp: bool = True
generate_ahead_steps: int = 3

# Performance Tuning
dataloader_prefetch_steps: int = 10
param_update_preduce_bucket_size_mb: int = 512
```

## Usage

### Basic Usage

```python
from verl.trainer.ppo.pipeline import AsyncTrainingFlow

# Initialize the training flow
flow = AsyncTrainingFlow(
    trainer=trainer,
    enable_async_rl=True,
    enable_param_sync=True,
    enable_async_mode=True
)

# Run the async training
await flow.run()
```

### Advanced Configuration

```python
# Custom state machine creation
from verl.trainer.ppo.pipeline import create_role_state_machine

state_machine = create_role_state_machine(
    role_name="train",
    pipeline=pipeline,
    trainer=trainer,
    use_async_rl=True
)
```

## Future Enhancements (TODO)

### 1. Validation Asynchronous Support
- **Status**: Currently disabled
- **Plan**: Add validation state machine
- **Integration**: Interleave with dataloader and generate flows
- **Goal**: Create parallel data streams for training and validation

### 2. Critic Asynchronous Support
- **Status**: Limited to GRPO support
- **Plan**: Extend to other algorithms
- **Goal**: Full critic component asynchrony

### 3. LogP Asynchronous Support
- **Status**: Partially implemented
- **Plan**: Complete logp to train recv+load operations
- **Goal**: Full logp component asynchrony

### 4. Off-Policy Monitoring
- **Status**: Not implemented
- **Plan**: Monitor off-policy lag steps
- **Goal**: Track param_update lag behind actor train-step
- **Metrics**: Monitor generate param_update vs actor train-step difference

## Technical Details

### Resource Management

The system implements sophisticated resource management through:
- **Resource Locking**: Prevents resource contention between components
- **Task Scheduling**: Intelligent task overlap and scheduling
- **Memory Management**: Efficient memory usage and cleanup

### Error Handling

- **State Machine Error Recovery**: Each state machine handles its own errors
- **Pipeline Resilience**: System continues operation even if individual components fail
- **Graceful Degradation**: Falls back to synchronous mode if needed

### Monitoring and Debugging

- **Timing Statistics**: Comprehensive timing collection for performance analysis
- **State Tracking**: Real-time state machine status monitoring
- **Performance Metrics**: Detailed performance and utilization metrics

## Contributing

When contributing to the async-RL pipeline:

1. **State Machine Design**: Follow the established state machine patterns
2. **Resource Management**: Ensure proper resource locking and cleanup
3. **Performance**: Consider the impact on overall pipeline performance
4. **Testing**: Test both synchronous and asynchronous modes
5. **Documentation**: Update this README for any new features

## References

- **Paper**: [Async-RL: Asynchronous Reinforcement Learning for Large-Scale Training]
- **Implementation**: Based on VERL framework
- **Performance**: Benchmarked on Red-MoE-16B model

---

For more detailed information about specific components, please refer to the individual module documentation. 