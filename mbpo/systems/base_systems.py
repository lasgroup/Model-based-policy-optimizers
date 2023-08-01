from abc import ABC
from typing import Generic

import chex
import jax.numpy as jnp
import jax.random

from mbpo.systems.dynamics.base_dynamics import Dynamics, DynamicsParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
import flax.struct as struct


@chex.dataclass
class SystemParams(Generic[DynamicsParams, RewardParams]):
    dynamics_params: DynamicsParams
    reward_params: RewardParams
    key: jax.random.PRNGKey = struct.field(default_factory=lambda: jax.random.PRNGKey(0))


@chex.dataclass
class SystemState(Generic[DynamicsParams, RewardParams]):
    x_next: chex.Array
    reward: chex.Array
    system_params: SystemParams[DynamicsParams, RewardParams]
    done: chex.Array = struct.field(default_factory=lambda: jnp.array(0, dtype=int))


class System(ABC, Generic[DynamicsParams, RewardParams]):
    def __init__(self, dynamics: Dynamics[DynamicsParams], reward: Reward[RewardParams]):
        self.dynamics = dynamics
        self.reward = reward
        self.x_dim = dynamics.x_dim
        self.u_dim = dynamics.u_dim

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
