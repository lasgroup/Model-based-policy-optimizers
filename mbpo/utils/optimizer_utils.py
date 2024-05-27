import jax
from functools import partial
from mbpo.systems import System, SystemParams
from mbpo.utils.type_aliases import OptimizerState
import jax.numpy as jnp
import chex
from brax.training.types import Transition
from typing import Callable


@partial(jax.jit, static_argnums=(0, 4))
def rollout_actions(
        system: System,
        system_params: SystemParams,
        init_state: chex.Array,
        actions: chex.Array,
        horizon: int,
) -> Transition:
    """
    system: System class to propagate the dynamics.
    system_params: Dynamics model, reward model params and sampling key
    init_state: initial state to optimize from
    action: sequence of actions.
    horizon: Horizon for rollout.
    """
    assert actions.shape[0] == horizon

    def step(carry, acs):
        obs = carry[0]
        sys_params = carry[-1]

        system_output = system.step(
            x=obs,
            u=acs,
            system_params=sys_params,
        )
        next_obs = system_output.x_next
        reward = system_output.reward
        next_sys_params = system_output.system_params
        carry = [next_obs, next_sys_params]
        outs = [next_obs, reward]
        return carry, outs

    ins = actions
    carry = [init_state, system_params]
    _, outs = jax.lax.scan(step, carry, ins, length=horizon)
    next_state = outs[0]
    state = jnp.zeros_like(next_state)
    state = state.at[0, ...].set(init_state)
    state = state.at[1:, ...].set(next_state[:-1, ...])
    rewards = outs[1]
    transitions = Transition(
        observation=state,
        action=actions,
        reward=rewards,
        next_observation=next_state,
        discount=jnp.ones_like(rewards),
    )
    return transitions


@partial(jax.jit, static_argnums=(0, 3, 5, 6))
def rollout_policy(
        system: System,
        system_params: SystemParams,
        init_state: chex.Array,
        policy: Callable,
        policy_state: OptimizerState,
        horizon: int,
        stop_grads: bool = True,
) -> Transition:
    """
    system: System class to propagate the dynamics.
    system_params: Dynamics model, reward model params and sampling key
    init_state: initial state to optimize from
    policy: Callable function that takes obs and parameters to return actions.
    policy_state: Stores the policy params.
    horizon: Horizon for rollout.
    """

    def step(carry, ins):
        obs = carry[0]
        sys_params = carry[1]
        policy_param = carry[-1]
        if stop_grads:
            acs, new_policy_state = policy(jax.lax.stop_gradient(obs), policy_param)
        else:
            acs, new_policy_state = policy(obs, policy_param)
        system_output = system.step(
            x=obs,
            u=acs,
            system_params=sys_params,
        )
        next_obs = system_output.x_next
        reward = system_output.reward
        next_sys_params = system_output.system_params
        carry = [next_obs, next_sys_params, new_policy_state]
        outs = [next_obs, acs, reward]
        return carry, outs

    carry = [init_state, system_params, policy_state]
    _, outs = jax.lax.scan(step, carry, xs=None, length=horizon)
    next_state = outs[0]
    state = jnp.zeros_like(next_state)
    state = state.at[0, ...].set(init_state)
    state = state.at[1:, ...].set(next_state[:-1, ...])
    actions = outs[1]
    rewards = outs[-1]
    transitions = Transition(
        observation=state,
        action=actions,
        reward=rewards,
        next_observation=next_state,
        discount=jnp.ones_like(rewards),
    )
    return transitions


@partial(jax.jit, static_argnums=(2, 3))
def lambda_return(reward: jax.Array,
                  next_values: jax.Array,
                  discount: float,
                  lambda_: float):
    """Taken from https://github.com/danijar/dreamer/"""
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert reward.ndim == next_values.ndim, (reward.shape, next_values.shape)
    inputs = reward + discount * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, inp: inp + discount * lambda_ * agg,
        inputs, next_values[-1], reverse=True)
    return returns


def static_scan(fn, inputs: jax.Array, start: jax.Array, reverse=False):
    """Taken from https://github.com/danijar/dreamer/"""
    if reverse:
        inputs = jax.tree_util.tree_map(lambda x: x[::-1], inputs)

    def step(carry, ins):
        x = carry
        inp = ins
        next = fn(x, inp)
        out = next
        carry = next
        return carry, out

    carry = start
    carry, outs = jax.lax.scan(step, carry, xs=inputs)
    if reverse:
        outs = jax.tree_util.tree_map(lambda x: x[::-1], outs)
    return outs


def soft_update(
        target_params, online_params, tau=0.005
):
    updated_params = jax.tree_util.tree_map(
        lambda old, new: (1 - tau) * old + tau * new, target_params, online_params
    )
    return updated_params
