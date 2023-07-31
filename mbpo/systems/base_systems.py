from abc import ABC, abstractmethod
from typing import Generic

import chex
import jax.numpy as jnp

from mbpo.systems.dynamics.base_dynamics import Dynamics, DynamicsParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams


@chex.dataclass
class SystemParams(Generic[DynamicsParams, RewardParams]):
    dynamics_params: DynamicsParams
    reward_params: RewardParams


@chex.dataclass
class SystemOutput(Generic[DynamicsParams, RewardParams]):
    x_next: chex.Array
    reward: chex.Array
    done: chex.Array = jnp.array(0, dtype=int)
    system_params: SystemParams[DynamicsParams, RewardParams]


class System(ABC, Generic[DynamicsParams, RewardParams]):
    def __init__(self, dynamics: Dynamics[DynamicsParams], reward: Reward[RewardParams], *args, **kwargs):
        self.dynamics = dynamics
        self.reward = reward

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, RewardParams],
             ) -> SystemOutput:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        pass

    @abstractmethod
    def reset(self, rng: jnp.ndarray) -> SystemOutput:
        pass
