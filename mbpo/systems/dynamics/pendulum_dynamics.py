from typing import Tuple

import chex
import distrax
import flax.struct as struct
import jax.numpy as jnp
from distrax import Distribution

from mbpo.systems.dynamics.base_dynamics import Dynamics


@chex.dataclass
class PendulumDynamicsParams:
    max_speed: chex.Array = struct.field(default_factory=lambda: jnp.array(8.0))
    max_torque: chex.Array = struct.field(default_factory=lambda: jnp.array(2.0))
    dt: chex.Array = struct.field(default_factory=lambda: jnp.array(0.05))
    g: chex.Array = struct.field(default_factory=lambda: jnp.array(9.81))
    m: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
    l: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))


class PendulumDynamics(Dynamics[PendulumDynamicsParams]):
    def __init__(self):
        super().__init__(x_dim=3, u_dim=1)

    def init_params(self, key: chex.PRNGKey) -> PendulumDynamicsParams:
        return PendulumDynamicsParams()

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: PendulumDynamicsParams) -> Tuple[Distribution, PendulumDynamicsParams]:
        chex.assert_shape(x, (self.x_dim,))
        chex.assert_shape(u, (self.u_dim,))
        th = jnp.arctan2(x[1], x[0])
        thdot = x[-1]
        dt = dynamics_params.dt
        x_compressed = jnp.array([th, thdot])
        dx = self.ode(x_compressed, u, dynamics_params)
        newth = th + dx[0] * dt
        newthdot = thdot + dx[-1] * dt
        newthdot = jnp.clip(newthdot, -dynamics_params.max_speed, dynamics_params.max_speed)
        mean = jnp.asarray([jnp.cos(newth), jnp.sin(newth), newthdot])
        mean = mean.reshape(self.x_dim)
        std = jnp.zeros_like(mean)
        next_state_dist = distrax.Normal(loc=mean, scale=std)
        return next_state_dist, dynamics_params

    def ode(self, x_compressed: chex.Array, u: chex.Array, dynamics_params: PendulumDynamicsParams) -> chex.Array:
        chex.assert_shape(x_compressed, (self.x_dim - 1,))
        chex.assert_shape(u, (self.u_dim,))
        thdot = x_compressed[-1]
        th = x_compressed[0]

        g = dynamics_params.g
        m = dynamics_params.m
        l = dynamics_params.l
        dt = dynamics_params.dt
        u = jnp.clip(u, -1, 1) * dynamics_params.max_torque
        newthddot = (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * u)
        newthdot = thdot + newthddot * dt
        newthdot = jnp.clip(newthdot, -dynamics_params.max_speed, dynamics_params.max_speed)
        return jnp.asarray([newthdot, newthddot])
