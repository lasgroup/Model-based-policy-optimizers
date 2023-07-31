import jax
from functools import partial
from mbpo.systems import System, SystemParams
import jax.numpy as jnp
import chex
from brax.training.types import Transition


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
