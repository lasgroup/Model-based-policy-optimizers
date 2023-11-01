from typing import Generic
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
from mbpo.systems.base_systems import SystemParams
import chex


@chex.dataclass
class OptimizerState(Generic[DynamicsParams, RewardParams]):
    system_params: SystemParams[DynamicsParams, RewardParams]


@chex.dataclass
class OptimizerTrainingOutPut(Generic[DynamicsParams, RewardParams]):
    optimizer_state: OptimizerState[DynamicsParams, RewardParams]
