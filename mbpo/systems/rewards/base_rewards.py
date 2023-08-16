from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

import chex
from distrax import Distribution

RewardParams = TypeVar('RewardParams')


class Reward(ABC, Generic[RewardParams]):
    def __init__(self, x_dim: int, u_dim: int):
        self.x_dim = x_dim
        self.u_dim = u_dim

    @abstractmethod
    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: RewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        pass

    @abstractmethod
    def init_params(self, key: chex.PRNGKey, kwargs: dict = None) -> RewardParams:
        pass
