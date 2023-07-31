import jax.random

from mbpo.systems.dynamics.pendulum_dynamics import PendulumDynamics, PendulumDynamicsParams
from mbpo.systems.rewards.pendulum_reward import PendulumReward, PendulumRewardParams
from mbpo.systems.base_systems import System, SystemOutput, SystemParams
import chex
import jax.numpy as jnp
from functools import partial


class PendulumSystem(System):
    def __init__(self):
        super().__init__(dynamics=PendulumDynamics(), reward=PendulumReward())
        self.min_action = -1.0
        self.max_action = 1.0

    @partial(jax.jit, static_argnums=0)
    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams,
             ) -> SystemOutput:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        x_nex_dist, new_dynamics_params = self.dynamics.next_state(x, u, system_params.dynamics_params)
        x_next = x_nex_dist.mean()
        reward_dist, new_reward_params = self.reward(x, u, system_params.reward_params, x_next)
        reward = reward_dist.mean()
        return SystemOutput(
            x_next=x_next,
            reward=reward,
            system_params=SystemParams(dynamics_params=new_dynamics_params, reward_params=new_reward_params),
        )

    def reset(self, rng: jnp.ndarray) -> SystemOutput:
        return SystemOutput(
            x_next=jnp.array([-1.0, 0.0, 0.0]),
            reward=jnp.array([0.0]).squeeze(),
            system_params=SystemParams(dynamics_params=PendulumDynamicsParams(), reward_params=PendulumRewardParams()),
        )


if __name__ == '__main__':
    import jax.random as random

    num_envs = 20
    key = random.PRNGKey(0)
    reset_keys = random.split(key, num_envs + 1)
    key = reset_keys[0]
    reset_keys = reset_keys[1:]
    system = PendulumSystem()
    system_state = jax.vmap(system.reset)(reset_keys)
    action_key, key = random.split(key, 2)
    actions = random.uniform(key=action_key, shape=(num_envs, 1))
    next_system_state = jax.vmap(system.step)(system_state.x_next, actions, system_state.system_params)
    chex.assert_shape(next_system_state.reward, (num_envs,))
    chex.assert_shape(next_system_state.x_next, (num_envs, 3))
