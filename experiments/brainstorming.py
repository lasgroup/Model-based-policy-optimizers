from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple, Dict, Any

from flax import struct
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
class SystemState(Generic[DynamicsParams, RewardParams]):
    x_next: chex.Array
    reward: chex.Array
    done: chex.Array = jnp.array(0, dtype=int)
    system_params: SystemParams[DynamicsParams, RewardParams]
    metrics: Dict[str, chex.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class Dynamics(ABC, Generic[DynamicsParams]):
    pass

    @abstractmethod
    def next_state(self,
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

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[DynamicsParams, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        pass

    @abstractmethod
    def reset(self, rng: jnp.ndarray) -> SystemState:
        pass
