from mbpo.systems import PendulumSystem

import jax
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


def test_prediction_dimension():
    assert next_system_state.x_next.shape == (num_envs, 3)
    assert next_system_state.reward.shape == (num_envs,)
