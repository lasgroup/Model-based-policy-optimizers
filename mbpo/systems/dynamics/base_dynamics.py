from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

import chex
from distrax import Distribution

DynamicsParams = TypeVar('DynamicsParams')


class Dynamics(ABC, Generic[DynamicsParams]):
    def __init__(self, x_dim: int, u_dim: int):
        self.x_dim = x_dim
        self.u_dim = u_dim

    @abstractmethod
    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        pass

    @abstractmethod
    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        pass
