from mbpo.systems.rewards.base_rewards import Reward, RewardParams
import chex
import jax.numpy as jnp
from typing import Tuple
from distrax import Distribution
import distrax
import flax.struct as struct


@chex.dataclass
class PendulumRewardParams:
    control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
    angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


class PendulumReward(Reward):

    def __init__(self):
        super().__init__()
        self.state_dim = 3
        self.input_dim = 1

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: PendulumRewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, PendulumRewardParams]:
        theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
        theta = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        angle_cost = reward_params.angle_cost
        target_angle = reward_params.target_angle
        control_cost = reward_params.control_cost
        reward = -(angle_cost * (theta - target_angle) ** 2 +
                   0.1 * omega ** 2) - control_cost * u ** 2
        reward = reward.squeeze()
        reward_dist = distrax.Normal(loc=reward, scale=jnp.zeros_like(reward))
        return reward_dist, reward_params
