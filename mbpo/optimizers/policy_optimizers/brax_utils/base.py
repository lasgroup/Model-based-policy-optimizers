from typing import Optional, Dict, Any

import chex
from brax import base
from flax import struct

from mbpo.systems import SystemParams
from mbpo.systems.dynamics.base_dynamics import DynamicsParams
from mbpo.systems.rewards.base_rewards import RewardParams


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
