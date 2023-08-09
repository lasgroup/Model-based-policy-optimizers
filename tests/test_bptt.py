import jax.numpy as jnp
import jax.random
import jax.random as jr
import jax.tree_util as jtu
from brax.training.replay_buffers import UniformSamplingQueue
from brax.training.types import Transition
from jax import jit
from jax.lax import scan

from mbpo.optimizers.policy_optimizers.bptt_optimizer import BPTTOptimizer
from mbpo.systems import PendulumSystem

system = PendulumSystem()
key = jax.random.PRNGKey(seed=0)
reset_key, key = jax.random.split(key, 2)
# Create replay buffer
init_sys_state = system.reset(rng=reset_key)
sample_key, key = jax.random.split(key, 2)
num_samples = 1


def sample_obs(key):
    samples = jax.random.uniform(key=key, shape=(2,), minval=-1, maxval=1)
    theta, angular_v = jnp.split(samples, 2, axis=-1)
    theta = jnp.ones_like(theta) * jnp.pi
    angular_v = jnp.zeros_like(angular_v) * 8.0
    new_state = jnp.concatenate([jnp.cos(theta), jnp.sin(theta), angular_v])
    return new_state


sample_key = jax.random.split(sample_key, num_samples)
obs = jax.vmap(sample_obs)(sample_key)

dummy_sample = Transition(observation=init_sys_state.x_next,
                          action=jnp.zeros(shape=(system.u_dim,)),
                          reward=init_sys_state.reward,
                          discount=jnp.array(0.99),
                          next_observation=init_sys_state.x_next)

sampling_buffer = UniformSamplingQueue(max_replay_size=10000,
                                       dummy_data_sample=dummy_sample,
                                       sample_batch_size=1)

sampling_buffer_state = sampling_buffer.init(jr.PRNGKey(0))
sample = Transition(observation=obs,
                    action=jnp.zeros(shape=(num_samples, system.u_dim)),
                    reward=jnp.zeros(num_samples),
                    discount=jnp.ones(num_samples),
                    next_observation=obs)
sampling_buffer_state = sampling_buffer.insert(sampling_buffer_state, sample)

optimizer = BPTTOptimizer(
    system=system,
    action_dim=1,
    obs_dim=3,
    horizon=20,
    num_samples_per_gradient_update=50,
    train_steps=1000,
    init_stddev=2.0,
    lambda_=0.97,
    critic_updates_per_policy_update=1,
    use_best_trained_policy=True,
)

output = optimizer.train(
    buffer_state=sampling_buffer_state,
    bptt_state=optimizer.init_state,
    system_params=init_sys_state.system_params,
)

bptt_state = output.bptt_state


def rollout_bptt(carry, ins):
    system_state, bptt_state = carry[0], carry[1]
    action, new_bptt_optimizer_state = optimizer.act(obs=system_state.x_next, opt_state=bptt_state)
    new_system_state = system.step(x=system_state.x_next, u=action,
                                   system_params=system_state.system_params)

    carry = [new_system_state, new_bptt_optimizer_state]
    outs = [new_system_state.x_next, new_system_state.reward]
    return carry, outs


carry = [init_sys_state, bptt_state]
carry, outs = scan(rollout_bptt, carry, xs=None, length=200)

rewards = outs[-1]

