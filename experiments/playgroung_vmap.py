from typing import Tuple

import chex
import jax.numpy as jnp
from jax import vmap


@chex.dataclass
class Params:
    dynamics: chex.Array
    policy: chex.Array


def norm(params: Params, x: chex.Array) -> Tuple[Params, chex.Array]:
    assert params.dynamics.ndim == params.policy.ndim == 2
    return Params(dynamics=jnp.linalg.norm(params.dynamics), policy=jnp.linalg.norm(params.policy)), jnp.linalg.norm(x)


v_norm = vmap(norm, in_axes=(Params(dynamics=0, policy=None), 0), out_axes=(Params(dynamics=0, policy=None), 0))

dynamics_params = jnp.stack([2 * jnp.ones((2, 2)), jnp.ones((2, 2)), jnp.ones((2, 2))])

v_parmas = Params(dynamics=dynamics_params, policy=jnp.ones((2, 2)))
v_x = jnp.ones((3, 2))

print(v_norm(v_parmas, v_x))
