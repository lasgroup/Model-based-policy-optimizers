from typing import NamedTuple

import chex
import jax.numpy as jnp


class Transition(NamedTuple):
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    discount: chex.Array
    next_observation: chex.Array
    truncation: chex.Array




@chex.dataclass
class SystemParams:
    a: chex.Array = jnp.array(0, dtype=int)
