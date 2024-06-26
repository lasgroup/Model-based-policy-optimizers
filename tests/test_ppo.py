import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax.lax import scan

from mbpo.optimizers import PPOOptimizer
from mbpo.systems import PendulumSystem

system = PendulumSystem()
# Create replay buffer
init_sys_state = system.reset(rng=jr.PRNGKey(0))

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
optimizer = PPOOptimizer(system=system,
                         true_buffer=sampling_buffer,
                         num_timesteps=1_000_000,
                         episode_length=200,
                         action_repeat=1,
                         num_envs=256,
                         num_eval_envs=1,
                         lr=3e-3,
                         wd=0,
                         entropy_cost=1e-1,
                         discounting=0.99,
                         seed=0,
                         unroll_length=40,
                         batch_size=128,
                         num_minibatches=32,
                         num_updates_per_batch=8,
                         num_evals=20,
                         normalize_observations=True,
                         reward_scaling=1,
                         clipping_epsilon=0.3,
                         gae_lambda=0.95,
                         deterministic_eval=True,
                         normalize_advantage=True,
                         policy_hidden_layer_sizes=(64, 64),
                         critic_hidden_layer_sizes=(64, 64),
                         wandb_logging=False,
                         )

# There is a tradeoff between num_envs, grad_updates_per_step and num_env_steps_between_updates
# grad_updates_per_step should be roughly equal to num_envs * num_env_steps_between_updates

init_optimizer_state = optimizer.init(key=jr.PRNGKey(0),
                                      true_buffer_state=sampling_buffer_state)

ppo_output = optimizer.train(opt_state=init_optimizer_state)


def policy(x):
    return optimizer.act(x, ppo_output.optimizer_state, evaluate=True)


def step(x, _):
    u = policy(x)[0]
    next_sys_state = system.step(x, u, ppo_output.optimizer_state.system_params)
    return next_sys_state.x_next, (x, u, next_sys_state.reward)


system_state_init = system.reset(rng=jr.PRNGKey(0))
x_init = system_state_init.x_next

horizon = 200
x_last, trajectory = scan(step, x_init, None, length=horizon)


def test_good_fit():
    assert ppo_output.summary[-1]['eval/episode_reward'] >= -400


def test_small_reward():
    assert jnp.abs(trajectory[2][-1]) <= 0.1
