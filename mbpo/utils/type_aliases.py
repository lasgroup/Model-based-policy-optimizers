from typing import Generic
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
from mbpo.systems.base_systems import SystemParams
from brax.training.replay_buffers import ReplayBufferState
import chex
import jax


@chex.dataclass
class OptimizerState(Generic[DynamicsParams, RewardParams]):
    true_buffer_state: ReplayBufferState
    system_params: SystemParams[DynamicsParams, RewardParams]
    key: jax.random.PRNGKey


@chex.dataclass
class OptimizerTrainingOutPut(Generic[DynamicsParams, RewardParams]):
    optimizer_state: OptimizerState[DynamicsParams, RewardParams]
