from datetime import datetime

import flax.linen as nn

import jax
import jax.random as jr
import matplotlib.pyplot as plt
from brax import envs
from jax.nn import squareplus, swish

from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

env_name = 'inverted_pendulum'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup',
# 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'positional'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

optimizer = SAC(
    environment=env,
    num_timesteps=20_000,
    episode_length=1000,
    action_repeat=1,
    num_env_steps_between_updates=10,
    num_envs=4,
    num_eval_envs=32,
    lr_alpha=3e-4,
    lr_policy=3e-4,
    lr_q=3e-4,
    wd_alpha=0.,
    wd_policy=0.,
    wd_q=0.,
    max_grad_norm=1e5,
    discounting=0.99,
    batch_size=32,
    num_evals=20,
    normalize_observations=True,
    reward_scaling=1.,
    tau=0.005,
    min_replay_size=10 ** 2,
    max_replay_size=10 ** 5,
    grad_updates_per_step=10 * 32,
    deterministic_eval=True,
    init_log_alpha=0.,
    policy_hidden_layer_sizes=(64, 64),
    policy_activation=swish,
    critic_hidden_layer_sizes=(64, 64),
    critic_activation=swish,
    wandb_logging=False,
    return_best_model=False,
)

xdata, ydata = [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()


optimizer.run_training(key=jr.PRNGKey(0), progress_fn=progress)

# wandb.finish()

# train_fn = {
#     'inverted_pendulum': functools.partial(ppo.train,
#                                            num_timesteps=2_000_000,
#                                            num_evals=20,
#                                            reward_scaling=10,
#                                            episode_length=1000,
#                                            normalize_observations=True,
#                                            action_repeat=1,
#                                            unroll_length=5,
#                                            num_minibatches=32,
#                                            num_updates_per_batch=4,
#                                            discounting=0.97,
#                                            learning_rate=3e-4,
#                                            entropy_cost=1e-2,
#                                            num_envs=2048,
#                                            batch_size=1024,
#                                            seed=1),
# }[env_name]
#


#
#
# make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
#

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
