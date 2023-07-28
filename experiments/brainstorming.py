from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

import chex
import jax.numpy as jnp
from distrax import Distribution

DynamicsParams = TypeVar('DynamicsParams')
RewardParams = TypeVar('RewardParams')


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


class Dynamics(ABC, Generic[DynamicsParams]):
    pass

    @abstractmethod
    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        pass


class Reward(ABC, Generic[RewardParams]):
    pass

    @abstractmethod
    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: RewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        pass


class System(ABC, Generic[DynamicsParams, RewardParams]):
    def __init__(self, dynamics: Dynamics[DynamicsParams], reward: Reward[RewardParams], *args, **kwargs):
        self.dynamics = dynamics
        self.reward = reward

    def __call__(self,
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
