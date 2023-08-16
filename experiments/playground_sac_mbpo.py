import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax.lax import scan

import wandb
from mbpo.optimizers.policy_optimizers.brax_optimizers import SACOptimizer
from mbpo.systems import PendulumSystem

system = PendulumSystem()
# Create replay buffer
init_sys_state = system.reset(rng=0)

dummy_sample = Transition(observation=init_sys_state.x_obs,
                          action=jnp.zeros(shape=(system.u_dim,)),
                          reward=init_sys_state.reward,
                          discount=jnp.array(0.99),
                          next_observation=init_sys_state.x_obs)

sampling_buffer = UniformSamplingQueue(max_replay_size=10,
                                       dummy_data_sample=dummy_sample,
                                       sample_batch_size=1)

sampling_buffer_state = sampling_buffer.init(jr.PRNGKey(0))
sampling_buffer_state = sampling_buffer.insert(sampling_buffer_state,
                                               jtu.tree_map(lambda x: x[None, ...], dummy_sample))

# Create MBPO environment
optimizer = SACOptimizer(system=system,
                         true_buffer=sampling_buffer,
                         dummy_true_buffer_state=sampling_buffer_state,
                         num_timesteps=20_000, num_evals=20, reward_scaling=1,
                         episode_length=200, normalize_observations=True, action_repeat=1,
                         discounting=0.99, lr_policy=3e-4, lr_alpha=3e-4, lr_q=3e-4, num_envs=32,
                         batch_size=64, grad_updates_per_step=20 * 32, max_replay_size=2 ** 14, min_replay_size=2 ** 7,
                         num_eval_envs=1,
                         deterministic_eval=True, tau=0.005, wd_policy=0, wd_q=0, wd_alpha=0, wandb_logging=True,
                         num_env_steps_between_updates=20, policy_hidden_layer_sizes=(128, 128, 128),
                         critic_hidden_layer_sizes=(128, 128, 128),
                         )

# There is a tradeoff between num_envs, grad_updates_per_step and num_env_steps_between_updates
# grad_updates_per_step should be roughly equal to num_envs * num_env_steps_between_updates

wandb.init(
    project="Pendulum test MBPO",
    group='test group',
)
init_optimizer_state = optimizer.init(key=jr.PRNGKey(0),
                                      true_buffer_state=sampling_buffer_state)

sac_output = optimizer.train(opt_state=init_optimizer_state)


def policy(x):
    return optimizer.act(x, sac_output.state, sac_output.state.system_params, evaluate=False)


def step(x, _):
    u = policy(x)[0]
    next_sys_state = system.step(x, u, sac_output.state.system_params)
    return next_sys_state.x_obs, (x, u, next_sys_state.reward)


system_state_init = system.reset(rng=jr.PRNGKey(0))
x_init = system_state_init.x_obs

horizon = 200
x_last, trajectory = scan(step, x_init, None, length=horizon)

plt.plot(trajectory[0], label='Xs')
plt.plot(trajectory[1], label='Us')
plt.plot(trajectory[2], label='Rewards')
plt.legend()
plt.show()
