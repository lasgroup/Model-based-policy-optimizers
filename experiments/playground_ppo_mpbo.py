import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax.lax import scan

import wandb
from mbpo.optimizers.policy_optimizers.model_based_ppo import ModelBasedPPO
from mbpo.systems import PendulumSystem

system = PendulumSystem()
# Create replay buffer
init_sys_state = system.reset(rng=0)

dummy_sample = Transition(observation=init_sys_state.x_next,
                          action=jnp.zeros(shape=(system.u_dim,)),
                          reward=init_sys_state.reward,
                          discount=jnp.array(0.99),
                          next_observation=init_sys_state.x_next)

sampling_buffer = UniformSamplingQueue(max_replay_size=10,
                                       dummy_data_sample=dummy_sample,
                                       sample_batch_size=1)

sampling_buffer_state = sampling_buffer.init(jr.PRNGKey(0))
sampling_buffer_state = sampling_buffer.insert(sampling_buffer_state,
                                               jtu.tree_map(lambda x: x[None, ...], dummy_sample))

# Create MBPO environment
optimizer = ModelBasedPPO(system=system,
                          true_buffer=sampling_buffer,
                          dummy_true_buffer_state=sampling_buffer_state,
                          num_timesteps=1_000_000,
                          episode_length=200,
                          action_repeat=1,
                          num_envs=16,
                          num_eval_envs=1,
                          lr=3e-3,
                          wd=0,
                          entropy_cost=1e-1,
                          discounting=0.99,
                          seed=0,
                          unroll_length=40,
                          batch_size=32,
                          num_minibatches=32,
                          num_updates_per_batch=4,
                          num_evals=20,
                          normalize_observations=True,
                          reward_scaling=1,
                          clipping_epsilon=0.3,
                          gae_lambda=0.95,
                          deterministic_eval=True,
                          normalize_advantage=True,
                          policy_hidden_layer_sizes=(128, 128, 128),
                          critic_hidden_layer_sizes=(128, 128, 128),
                          wandb_logging=True,
                          )

# There is a tradeoff between num_envs, grad_updates_per_step and num_env_steps_between_updates
# grad_updates_per_step should be roughly equal to num_envs * num_env_steps_between_updates

wandb.init(
    project="Pendulum test MBPO",
    group='test group',
)
init_optimizer_state = optimizer.init(key=jr.PRNGKey(0),
                                      true_buffer_state=sampling_buffer_state)

final_opt_state, metrics = optimizer.train(opt_state=init_optimizer_state)


def policy(x):
    return optimizer.act(x, final_opt_state, final_opt_state.system_params, evaluate=False)


def step(x, _):
    u = policy(x)[0]
    next_sys_state = system.step(x, u, final_opt_state.system_params)
    return next_sys_state.x_next, (x, u, next_sys_state.reward)


system_state_init = system.reset(rng=jr.PRNGKey(0))
x_init = system_state_init.x_next

horizon = 200
x_last, trajectory = scan(step, x_init, None, length=horizon)

plt.plot(trajectory[0], label='Xs')
plt.plot(trajectory[1], label='Us')
plt.plot(trajectory[2], label='Rewards')
plt.legend()
plt.show()
