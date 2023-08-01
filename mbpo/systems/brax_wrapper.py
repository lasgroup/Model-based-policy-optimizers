from typing import Any, Dict, Optional

import chex
from brax import base
from brax import envs
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from flax import struct
import jax.numpy as jnp
import jax.random as jr

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
                 sample_buffer_state: ReplayBufferState,
                 sample_buffer: UniformSamplingQueue):
        self.system = system
        self.sample_buffer_state = sample_buffer_state
        self.sample_buffer = sample_buffer

    def reset(self, rng: chex.Array) -> State:
        keys = jr.split(rng, 2)
        cur_buffer_state = self.sample_buffer_state.replace(key=keys[0])
        _, sample = self.sample_buffer.sample(cur_buffer_state)
        # Todo: here we need to sync the sample to the new state
        #       the only thing that we sample is the observation
        init_system_params = self.system.init_params(keys[1])
        reward, done = jnp.array(0.0), jnp.array(0.0)
        new_state = State(pipeline_state=None,
                          obs=sample.obs,
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

    def action_size(self) -> int:
        return self.system.u_dim

    def observation_size(self) -> int:
        return self.system.x_dim

    def backend(self) -> str:
        return 'string'
