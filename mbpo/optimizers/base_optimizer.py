from abc import ABC, abstractmethod
from typing import Generic, Tuple
from mbpo.utils.type_aliases import OptimizerState, OptimizerTrainingOutPut
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
import chex
from mbpo.systems.base_systems import System, SystemParams


class BaseOptimizer(ABC, Generic[RewardParams, DynamicsParams]):
    def __init__(self, system: System | None = None):
        self.system = system
        pass

    def set_system(self, system: System):
        self.system = system

    @abstractmethod
    def act(self, obs: chex.Array, opt_state: OptimizerState[RewardParams, DynamicsParams], evaluate: bool = True) ->\
            Tuple[chex.Array, OptimizerState]:
        pass

    def train(self, opt_state: OptimizerState[RewardParams, DynamicsParams], *args, **kwargs) \
            -> OptimizerTrainingOutPut[RewardParams, DynamicsParams]:
        return OptimizerTrainingOutPut(optimizer_state=opt_state)

    def init(self,
             key: chex.PRNGKey) -> OptimizerState:
        pass
