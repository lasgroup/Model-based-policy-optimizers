from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

import chex
from distrax import Distribution

DynamicsParams = TypeVar('DynamicsParams')


class Dynamics(ABC, Generic[DynamicsParams]):
    pass

    @abstractmethod
    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        pass
