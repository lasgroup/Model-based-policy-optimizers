from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Tuple

import chex
from distrax import Distribution

RewardParams = TypeVar('RewardParams')


class Reward(ABC, Generic[RewardParams]):
    pass

    @abstractmethod
    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: RewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        pass
