from datetime import datetime

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax import jit

import wandb
from mbpo.optimizers.sac_optimizer.sac import SAC
from mbpo.systems import PendulumSystem
from mbpo.systems.brax_wrapper import BraxWrapper

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

# Create brax environment
env = BraxWrapper(system=system,
                  sample_buffer_state=sampling_buffer_state,
                  sample_buffer=sampling_buffer)

state = jit(env.reset)(rng=jr.PRNGKey(0))

# There is a tradeoff between num_envs, grad_updates_per_step and num_env_steps_between_updates
# grad_updates_per_step should be roughly equal to num_envs * num_env_steps_between_updates

wandb.init(
    project="Pendulum test MBPO",
    group='test group',
)

sac_trainer = SAC(
    environment=env,
    num_timesteps=30_000, num_evals=20, reward_scaling=1,
    episode_length=200, normalize_observations=True, action_repeat=1,
    discounting=0.99, lr_policy=3e-4, lr_alpha=3e-4, lr_q=3e-4, num_envs=32,
    batch_size=64, grad_updates_per_step=20 * 32, max_replay_size=2 ** 14, min_replay_size=2 ** 7, num_eval_envs=1,
    deterministic_eval=True, tau=0.005, wd_policy=0, wd_q=0, wd_alpha=0, wandb_logging=True,
    num_env_steps_between_updates=20, policy_hidden_layer_sizes=(128, 128, 128),
    critic_hidden_layer_sizes=(128, 128, 128),
)

max_y = 0
min_y = -100

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    plt.xlim([0, sac_trainer.num_timesteps])
    # plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()


make_inference_fn, params, metrics = sac_trainer.run_training(key=jr.PRNGKey(0), progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')


def test_sac_good_fit():
    assert metrics['eval/episode_reward'] >= -320