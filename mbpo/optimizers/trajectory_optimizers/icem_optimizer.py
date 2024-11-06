from abc import abstractmethod
from functools import partial
from typing import Tuple, NamedTuple, Generic

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from jax import vmap, jit
from jax.nn import relu
from jax.numpy import sqrt, newaxis
from jax.numpy.fft import irfft, rfftfreq
from jaxtyping import Float, Array, Key, Scalar

from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.base_systems import System, SystemParams
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.utils.optimizer_utils import rollout_actions
from mbpo.utils.general_utils import powerlaw_psd_gaussian
from mbpo.utils.type_aliases import OptimizerState


class iCemParams(NamedTuple):
    """
    num_particles: int = 10
    num_samples: int = 500
    num_elites: int = 50
    init_std: float = initial std of the samples
    alpha: float = how softly we update the mean and var of elites to the next iteration
    num_steps: int = how many steps of samping we do
    exponent: float = How colored noise we want - higher the more correlated are the samples
    elite_set_fraction: float = Share of elites we take to the next iteration
    u_min: float | chex.Array = minimal value for action
    u_max: float | chex.Array = maximal value for action
    warm_start: bool = If we shift the action sequence for one and repeat the last action at initialization
    """
    num_particles: int = 10
    num_samples: int = 500
    num_elites: int = 50
    init_std: float = 0.5
    alpha: float = 0.0
    num_steps: int = 5
    exponent: float = 0.0
    elite_set_fraction: float = 0.3
    u_min: float | chex.Array = -1.0
    u_max: float | chex.Array = 1.0
    warm_start: bool = True
    lambda_constraint: float = 1e4


class ICemCarry(NamedTuple):
    key: Key[Array, '2']
    mean: Float[Array, 'horizon action_dim']
    std: Float[Array, 'horizon action_dim']
    best_value: Scalar
    best_sequence: Float[Array, 'horizon action_dim']
    prev_elites: Float[Array, 'num_elites horizon action_dim']


@chex.dataclass
class iCemOptimizerState(OptimizerState, Generic[DynamicsParams, RewardParams]):
    best_sequence: chex.Array
    best_reward: chex.Array

    @property
    def action(self):
        return self.best_sequence[0]


class AbstractCost:

    def __init__(self, horizon: int):
        self.horizon = horizon

    @abstractmethod
    def __call__(self,
                 states: Float[Array, 'horizon observation_dim'],
                 actions: Float[Array, 'horizon action_dim'],
                 ) -> Scalar:
        # This return the cost of the trajectory, i.e., \sum_{t=0}^{T-1}c(\vx_t, \vu_t)
        # We want the cost to be \E\left[\sum_{t=0}^{T-1}c(\vx_t, \vu_t)\right] <= 0
        pass


class iCemTO(BaseOptimizer, Generic[DynamicsParams, RewardParams]):
    def __init__(self,
                 horizon: int,
                 action_dim: int,
                 key: chex.PRNGKey = jax.random.PRNGKey(0),
                 opt_params: iCemParams = iCemParams(),
                 cost_fn: AbstractCost | None = None,
                 use_optimism: bool = True,
                 use_pessimism: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon
        self.opt_params = opt_params
        self.key = key
        self.opt_dim = (horizon,) + (action_dim,)
        self.action_dim = action_dim
        self.horizon = horizon
        self.cost_fn = cost_fn
        if use_optimism:
            self.summarize_raw_samples = jnp.max
        else:
            self.summarize_raw_samples = jnp.mean
        if use_pessimism:
            self.summarize_cost_samples = jnp.max
        else:
            self.summarize_cost_samples = jnp.mean

    def init(self, key: chex.Array) -> iCemOptimizerState:
        assert self.system is not None, "iCem optimizer requires system to be defined."
        init_key, dummy_buffer_key, key = jax.random.split(key, 3)
        system_params = self.system.init_params(init_key)
        dummy_buffer_state = self.dummy_true_buffer_state(dummy_buffer_key)
        return iCemOptimizerState(
            true_buffer_state=dummy_buffer_state,
            system_params=system_params,
            best_sequence=jnp.zeros(self.opt_dim),
            best_reward=jnp.zeros(1).squeeze(),
            key=key,
        )

    @partial(jax.jit, static_argnums=0)
    def optimize(
            self,
            initial_state: Float[Array, 'observation_dim'],
            opt_state: iCemOptimizerState,
    ):
        assert self.system is not None, "iCem optimizer requires system to be defined."

        # To estimate mean trajectory under some action sequence we sample self.opt_params.num_particles number of
        # noisy realization of the dynamics propagation
        def objective(seq: Float[Array, 'horizon action_dim'], key: Key[Array, '2']) -> Scalar:

            def optimize_fn(init_state: Float[Array, 'observation_dim'], rng: Key[Array, '2']):
                system_params = opt_state.system_params.replace(key=rng)
                return rollout_actions(system=self.system,
                                       system_params=system_params,
                                       init_state=init_state,
                                       horizon=self.horizon,
                                       actions=seq,
                                       )

            particles_rng = jr.split(key, self.opt_params.num_particles)
            transitions = jax.vmap(optimize_fn, in_axes=(None, 0))(initial_state, particles_rng)
            cost = 0

            # We summarize cost with mean or max (if optimism is true)
            reward = self.summarize_raw_samples(jnp.mean(transitions.reward, axis=-1))
            if self.cost_fn is not None:
                cost = vmap(self.cost_fn)(transitions.observation, transitions.action)
                assert cost.shape == (self.opt_params.num_particles,)
                # We summarize cost with mean or max (if pessimism is true)
                cost = self.summarize_cost_samples(cost)
            return reward - self.opt_params.lambda_constraint * relu(cost)

        get_best_action = lambda best_val, best_seq, val, seq: [val[-1], seq[-1]]
        get_curr_best_action = lambda best_val, best_seq, val, seq: [best_val, best_seq]
        num_prev_elites_per_iter = max(int(self.opt_params.elite_set_fraction * self.opt_params.num_elites), 1)

        def step(carry: ICemCarry, ins):
            # Split the key
            sampling_rng, particles_rng = jax.random.split(carry.key)
            sampling_rng = jax.random.split(key=sampling_rng, num=self.opt_params.num_samples + 1)
            key, sampling_rng = sampling_rng[0], sampling_rng[1:]
            particles_rng = jr.split(particles_rng, self.opt_params.num_samples + num_prev_elites_per_iter)

            # We create colored samples from gaussian of size (num_samples, np.prod(self.opt_dim))
            sampling_rng = vmap(lambda x: jr.split(x, self.action_dim))(sampling_rng)

            def colored_sample_fn(rng):
                return powerlaw_psd_gaussian(exponent=self.opt_params.exponent, size=self.horizon, rng=rng)

            colored_samples_all_dims_fn = vmap(colored_sample_fn, out_axes=1)
            colored_samples = vmap(colored_samples_all_dims_fn)(sampling_rng)
            assert colored_samples.shape == (self.opt_params.num_samples, self.horizon, self.action_dim)

            # Add noise, clip to [u_min, u_max], and reshape back
            action_samples = carry.mean + colored_samples * carry.std
            action_samples = jnp.clip(action_samples, a_max=self.opt_params.u_max, a_min=self.opt_params.u_min)
            action_samples = jnp.concatenate([action_samples, prev_elites], axis=0)

            # Calculate objective for all the samples
            values = jax.vmap(objective)(action_samples, particles_rng)
            assert values.shape == (self.opt_params.num_samples + num_prev_elites_per_iter,)

            # Prepare indices of elite samples (i.e. samples with the highest reward)
            best_elite_idx = np.argsort(values, axis=0)[-self.opt_params.num_elites:]

            # Take elite actions and their values
            elites = action_samples[best_elite_idx]
            elite_values = values[best_elite_idx]

            # Compute mean and var of elites actions
            elite_mean = jnp.mean(elites, axis=0)
            elite_var = jnp.var(elites, axis=0)

            # Do soft update of the mean and var
            mean = carry.mean * self.opt_params.alpha + (1 - self.opt_params.alpha) * elite_mean
            var = jnp.square(carry.std) * self.opt_params.alpha + (1 - self.opt_params.alpha) * elite_var

            # Compute std of the soft updated elites actions
            std = jnp.sqrt(var)

            # Find the best action so far
            best_elite = elite_values[-1]
            bests = jax.lax.cond(carry.best_value <= best_elite,
                                 get_best_action,
                                 get_curr_best_action,
                                 carry.best_value,
                                 carry.best_sequence,
                                 elite_values,
                                 elites)
            best_val, best_seq = bests[0], bests[-1]
            outs = [best_val, best_seq]

            # Take only num_prev_elites_per_iter elites to the next iteration
            elite_set = elites[-num_prev_elites_per_iter:]

            carry = ICemCarry(key=key, mean=mean, std=std, best_value=best_val, best_sequence=best_seq,
                              prev_elites=elite_set)
            return carry, outs

        best_value = -jnp.inf
        mean = jnp.zeros(self.opt_dim)

        # If we warm start the optimization we shift the action sequence for one and repeat the last action
        if self.opt_params.warm_start:
            mean = mean.at[:-1].set(opt_state.best_sequence[1:])
            mean = mean.at[-1].set(opt_state.best_sequence[-1])

        std = jnp.ones(self.opt_dim) * self.opt_params.init_std
        best_sequence = mean
        prev_elites = jnp.zeros((num_prev_elites_per_iter,) + self.opt_dim)
        optimizer_key, key = jax.random.split(opt_state.key, 2)
        new_opt_state = opt_state.replace(key=key)
        carry = ICemCarry(key=optimizer_key, mean=mean, std=std, best_value=best_value, best_sequence=best_sequence,
                          prev_elites=prev_elites)
        carry, outs = jax.lax.scan(step, carry, xs=None, length=self.opt_params.num_steps)
        new_opt_state = new_opt_state.replace(best_sequence=outs[1][-1, ...], best_reward=outs[0][-1, ...])
        return new_opt_state

    @partial(jax.jit, static_argnums=0)
    def act(self, obs: chex.Array, opt_state: iCemOptimizerState, evaluate: bool = True):
        new_opt_state = self.optimize(initial_state=obs, opt_state=opt_state)
        return new_opt_state.action, new_opt_state
    

class iCEMOptimizer(BaseOptimizer):
    """
    iCEM Wrapper to ensure consistency with SAC and PPO optimizers
    """
    def __init__(self,
                 horizon: int,
                 opt_params: iCemParams = iCemParams(),
                 system: System | None = None,
                 key: jr.PRNGKey = jr.PRNGKey(0),
                 **agent_kwargs):
        super().__init__(system, key)
        self.horizon = horizon
        self.key = key
        self.opt_params = opt_params
        self.agent_class = iCemTO
        self.agent_kwargs = agent_kwargs
        if system is not None:
            self.set_system(system)

    def set_system(self, system: System):
        super().set_system(system)

    def init(self,
             key: chex.PRNGKey,
             true_buffer_state = None) -> iCemOptimizerState:
        assert self.system is not None, "iCEM optimizer requires system to be defined."
        self.agent = self.agent_class(horizon=self.horizon,
                                      action_dim=self.system.u_dim,
                                      key = self.key,
                                      opt_params=self.opt_params,
                                      **self.agent_kwargs)
        self.agent.set_system(self.system)
        
        # true_buffer_state is used to ensure compatibility with SAC and PPO
        if true_buffer_state is None:
            dummy_buffer_key, key = jr.split(key, 2)
            true_buffer_state = self.dummy_true_buffer_state(dummy_buffer_key)
        agent_state = self.agent.init(key)
        agent_state.true_buffer_state = true_buffer_state
        return agent_state

    @partial(jit, static_argnums=(0, 3))
    def act(self,
            obs: chex.Array,
            opt_state: iCemOptimizerState,
            evaluate: bool = True) -> Tuple[chex.Array, iCemOptimizerState]:
        assert self.system is not None, "iCEM optimizer requires system to be defined."
        action, opt_state = self.agent.act(obs.reshape(-1,), opt_state, evaluate)
        return action.reshape(1, -1), opt_state


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    exponent_list = [0.0, 0.25, 1.0, 2.0, 3.0, 5.0, 10.0]
    fig, axs = plt.subplots(len(exponent_list))
    fig.suptitle('Colored noise with varying correlation coefficients')
    for i, exponent in enumerate(exponent_list):
        rng = jax.random.PRNGKey(seed=0)
        size = 10000
        samples = powerlaw_psd_gaussian(
            exponent=exponent,
            size=size,
            rng=rng,
        )
        x = np.arange(0, size)
        y = np.asarray(samples)
        axs[i].plot(x, y, label="coefficient " + str(exponent))
        axs[i].legend()
    plt.show()
    