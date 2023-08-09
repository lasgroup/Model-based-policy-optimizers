from typing import Any, Dict, Optional

import chex
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from brax import base
from brax import envs
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from brax.training.types import Transition
from flax import struct

from mbpo.systems.base_systems import System
from mbpo.systems.base_systems import SystemParams, DynamicsParams, RewardParams


@chex.dataclass
class State:
    """ Environment state for training and inference.
        We create a new state so that we can carry also system parameters."""

    pipeline_state: Optional[base.State]
    obs: chex.Array
    reward: chex.Array
    done: chex.Array
    system_params: SystemParams[DynamicsParams, RewardParams]
    metrics: Dict[str, chex.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class BraxWrapper(envs.Env):
    def __init__(self,
                 system: System,
                 system_params: SystemParams,
                 sample_buffer_state: ReplayBufferState,
                 sample_buffer: UniformSamplingQueue):
        self.system = system
        self.sample_buffer_state = sample_buffer_state
        self.sample_buffer = sample_buffer
        self.init_system_params = system_params

    def reset(self, rng: chex.Array) -> State:
        keys = jr.split(rng, 2)
        cur_buffer_state = self.sample_buffer_state.replace(key=keys[0])
        sample: Transition
        _, sample = self.sample_buffer.sample(cur_buffer_state)
        sample = jtu.tree_map(lambda x: x[0], sample)
        init_system_params = self.init_system_params
        reward, done = sample.reward, jnp.array(0.0)
        new_state = State(pipeline_state=None,
                          obs=sample.observation,
                          reward=reward,
                          done=done,
                          system_params=init_system_params)
        return new_state

    def step(self, state: State, action: chex.Array) -> State:
        next_sys_state = self.system.step(state.obs, action, state.system_params)
        next_obs = next_sys_state.x_next
        next_reward = next_sys_state.reward
        next_done = next_sys_state.done
        next_sys_params = next_sys_state.system_params
        next_state = state.replace(obs=next_obs,
                                   reward=next_reward,
                                   done=next_done,
                                   system_params=next_sys_params)
        return next_state

    @property
    def action_size(self) -> int:
        return self.system.u_dim

    @property
    def observation_size(self) -> int:
        return self.system.x_dim

    @property
    def backend(self) -> str:
        return 'string'
