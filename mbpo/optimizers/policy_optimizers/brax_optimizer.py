from typing import Tuple, List

import chex
import jax.random as jr
from brax.training import types
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from jaxtyping import PyTree

from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems.base_systems import System, SystemParams
from mbpo.systems.brax_wrapper import BraxWrapper


@chex.dataclass
class BraxState:
    system_params: SystemParams
    true_buffer_state: ReplayBufferState
    policy_params: PyTree
    key: chex.PRNGKey


@chex.dataclass
class BraxOutput:
    state: BraxState
    summary: List[types.Metrics]


class BraxOptimizer(BaseOptimizer[BraxState, BraxOutput]):
    def __init__(self,
                 agent_class,
                 system: System,
                 true_buffer: UniformSamplingQueue,
                 dummy_true_buffer_state: ReplayBufferState,
                 **agent_kwargs):
        super().__init__(system)
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.true_buffer = true_buffer
        self.dummy_true_buffer_state = dummy_true_buffer_state
        dummy_env = BraxWrapper(system=self.system,
                                system_params=self.system.init_params(jr.PRNGKey(0)),
                                sample_buffer_state=dummy_true_buffer_state,
                                sample_buffer=self.true_buffer)
        self.dummy_trainer = self.agent_class(environment=dummy_env, **self.agent_kwargs)
        self.make_policy = self.dummy_trainer.make_policy

    def init(self,
             key: chex.PRNGKey,
             true_buffer_state: ReplayBufferState) -> BraxState:
        keys = jr.split(key, 3)
        system_params = self.system.init_params(keys[0])
        training_state = self.dummy_trainer.init_training_state(keys[1])
        return BraxState(system_params=system_params,
                         true_buffer_state=true_buffer_state,
                         policy_params=training_state.get_policy_params(),
                         key=keys[2])

    def act(self,
            obs: chex.Array,
            opt_state: BraxState,
            system_params: SystemParams,
            evaluate: bool = True) -> Tuple[chex.Array, BraxState]:
        policy = self.make_policy(opt_state.policy_params, evaluate)
        key, subkey = jr.split(opt_state.key)
        action = policy(obs, subkey)[0]
        return action, opt_state.replace(key=key)

    def train(self,
              opt_state: BraxState,
              *args,
              **kwargs) -> BraxOutput:
        env = BraxWrapper(system=self.system,
                          system_params=opt_state.system_params,
                          sample_buffer_state=opt_state.true_buffer_state,
                          sample_buffer=self.true_buffer)

        trainer = self.agent_class(environment=env, **self.agent_kwargs)
        key, new_key = jr.split(opt_state.key)
        policy_params, metrics = trainer.run_training(key=new_key)
        new_opt_state = opt_state.replace(policy_params=policy_params, key=new_key)
        return BraxOutput(state=new_opt_state, summary=metrics)
