from datetime import datetime

import jax
import jax.random as jr
import matplotlib.pyplot as plt
from brax import envs
from jax.nn import swish

from mbpo.optimizers.policy_optimizers.ppo.ppo_brax_env import PPO

env_name = 'inverted_pendulum'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup',
# 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'positional'  # @param ['generalized', 'positional', 'spring']

env = envs.get_environment(env_name=env_name,
                           backend=backend)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

optimizer = PPO(
    environment=env,
    num_timesteps=2_000_000,
    num_evals=20,
    reward_scaling=10,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    lr=3e-4,
    entropy_cost=1e-2,
    num_envs=2048,
    batch_size=1024,
    seed=1,
    policy_hidden_layer_sizes=(32,) * 4,
    policy_activation=swish,
    critic_hidden_layer_sizes=(256,) * 5,
    critic_activation=swish,
    wandb_logging=False,
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
