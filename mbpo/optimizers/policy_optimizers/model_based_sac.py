from typing import Tuple, List

import chex
import jax.random as jr
from brax.training import types
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from jaxtyping import PyTree

from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.optimizers.policy_optimizers.sac_optimizer.sac import SAC
from mbpo.systems.base_systems import System, SystemParams
from mbpo.systems.brax_wrapper import BraxWrapper


@chex.dataclass
class SACMBState:
    system_params: SystemParams
    true_buffer_state: ReplayBufferState
    policy_params: PyTree
    key: chex.PRNGKey


@chex.dataclass
class SACOutput:
    sac_state: SACMBState
    sac_summary: List[types.Metrics]


class ModelBasedSac(BaseOptimizer[SACMBState, SACMBState]):
    def __init__(self,
                 system: System,
                 true_buffer: UniformSamplingQueue,
                 dummy_true_buffer_state: ReplayBufferState,
                 **sac_kwargs):
        super().__init__(system)
        self.sac_kwargs = sac_kwargs
        self.true_buffer = true_buffer
        self.dummy_true_buffer_state = dummy_true_buffer_state
        dummy_env = BraxWrapper(system=self.system,
                                system_params=self.system.init_params(jr.PRNGKey(0)),
                                sample_buffer_state=dummy_true_buffer_state,
                                sample_buffer=self.true_buffer)
        self.dummy_sac_trainer = SAC(environment=dummy_env, **self.sac_kwargs)
        self.make_policy = self.dummy_sac_trainer.make_policy

    def init(self,
             key: chex.PRNGKey,
             true_buffer_state: ReplayBufferState) -> SACMBState:
        keys = jr.split(key, 3)
        system_params = self.system.init_params(keys[0])
        sac_training_state = self.dummy_sac_trainer.init_training_state(keys[1])
        return SACMBState(system_params=system_params,
                          true_buffer_state=true_buffer_state,
                          policy_params=(sac_training_state.normalizer_params, sac_training_state.policy_params),
                          key=keys[2])

    def act(self,
            obs: chex.Array,
            opt_state: SACMBState,
            system_params: SystemParams,
            evaluate: bool = True) -> Tuple[chex.Array, SACMBState]:
        policy = self.make_policy(opt_state.policy_params, evaluate)
        # TODO: key should be passed to act?
        key, subkey = jr.split(opt_state.key)
        action = policy(obs, subkey)[0]
        return action, opt_state.replace(key=key)

    def train(self,
              opt_state: SACMBState,
              *args,
              **kwargs) -> SACOutput:
        env = BraxWrapper(system=self.system,
                          system_params=opt_state.system_params,
                          sample_buffer_state=opt_state.true_buffer_state,
                          sample_buffer=self.true_buffer)

        sac_trainer = SAC(environment=env, **self.sac_kwargs)
        key, new_key = jr.split(opt_state.key)
        policy_params, metrics = sac_trainer.run_training(key=new_key)
        new_opt_state = opt_state.replace(policy_params=policy_params, key=new_key)
        return SACOutput(sac_state=new_opt_state, sac_summary=metrics)
