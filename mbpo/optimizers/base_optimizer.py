from abc import ABC, abstractmethod
from typing import Generic, Tuple
from mbpo.utils.type_aliases import OptimizerState, OptimizerTrainingOutPut
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
import chex
from mbpo.systems.base_systems import System
import jax.numpy as jnp
import jax.random as jr
from brax.training.replay_buffers import ReplayBufferState, UniformSamplingQueue
from brax.training.types import Transition


class BaseOptimizer(ABC, Generic[RewardParams, DynamicsParams]):
    def __init__(self, system: System | None = None, key: jr.PRNGKey = jr.PRNGKey(0)):
        self.system = system
        self.key = key
        pass

    def set_system(self, system: System):
        self.system = system

    @abstractmethod
    def act(self, obs: chex.Array, opt_state: OptimizerState[RewardParams, DynamicsParams], evaluate: bool = True) -> \
            Tuple[chex.Array, OptimizerState]:
        pass

    def train(self, opt_state: OptimizerState[RewardParams, DynamicsParams]) \
            -> OptimizerTrainingOutPut[RewardParams, DynamicsParams]:
        return OptimizerTrainingOutPut(optimizer_state=opt_state)

    def init(self,
             key: chex.PRNGKey) -> OptimizerState:
        pass

    def dummy_true_buffer_state(self, key: chex.Array) -> ReplayBufferState:
        assert self.system is not None, "Base optimizer requires system to be defined."
        dummy_transition = Transition(
            observation=jnp.zeros(self.system.x_dim),
            action=jnp.zeros(self.system.u_dim),
            next_observation=jnp.zeros(self.system.x_dim),
            reward=jnp.zeros(1),
            discount=jnp.zeros(1),
        )
        sampling_buffer = UniformSamplingQueue(max_replay_size=10,
                                               dummy_data_sample=dummy_transition,
                                               sample_batch_size=1)
        return sampling_buffer.init(key)
