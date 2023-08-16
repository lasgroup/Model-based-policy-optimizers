from abc import ABC
from typing import Generic

import chex
import flax.struct as struct
import jax.numpy as jnp
import jax.random as jr

from mbpo.systems.dynamics.base_dynamics import Dynamics, DynamicsParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams


@chex.dataclass
class SystemParams(Generic[DynamicsParams, RewardParams]):
    dynamics_params: DynamicsParams
    reward_params: RewardParams
    key: chex.PRNGKey = struct.field(default_factory=lambda: jr.PRNGKey(0))


@chex.dataclass
class SystemState(Generic[DynamicsParams, RewardParams]):
    x_next: chex.Array
    reward: chex.Array
    system_params: SystemParams[DynamicsParams, RewardParams]
    done: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


class System(ABC, Generic[DynamicsParams, RewardParams]):
    def __init__(self, dynamics: Dynamics[DynamicsParams], reward: Reward[RewardParams]):
        self.dynamics = dynamics
        self.reward = reward
        self.x_dim = dynamics.x_dim
        self.u_dim = dynamics.u_dim
        # Here we have to set the axes of the system parameters which we want to vmap over

    @staticmethod
    def system_params_vmap_axes(axes: int = 0):
        return SystemParams(dynamics_params=None, reward_params=None, key=axes)

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

    def init_params(self,
                    key: chex.PRNGKey,
                    dynamics_kwargs: dict = None,
                    reward_kwargs: dict = None,) -> SystemParams[DynamicsParams, RewardParams]:
        keys = jr.split(key, 3)
        return SystemParams(
            dynamics_params=self.dynamics.init_params(key=keys[0],
                                                      kwargs=dynamics_kwargs),
            reward_params=self.reward.init_params(key=keys[1],
                                                  kwargs=reward_kwargs),
            key=keys[2],
        )
