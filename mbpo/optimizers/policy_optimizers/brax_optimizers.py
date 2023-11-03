from functools import partial
from typing import Tuple, List, Generic

import chex
import jax.random as jr
from brax.training import types
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from jax import jit
from jaxtyping import PyTree

from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.optimizers.policy_optimizers.ppo.ppo import PPO
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.base_systems import System
from mbpo.systems.brax_wrapper import BraxWrapper
from mbpo.utils.type_aliases import OptimizerState, OptimizerTrainingOutPut
from mbpo.systems.rewards.base_rewards import RewardParams
from mbpo.systems.dynamics.base_dynamics import DynamicsParams


@chex.dataclass
class BraxState(OptimizerState, Generic[DynamicsParams, RewardParams]):
    policy_params: PyTree


@chex.dataclass
class BraxOutput(OptimizerTrainingOutPut, Generic[DynamicsParams, RewardParams]):
    optimizer_state: BraxState[DynamicsParams, RewardParams]
    summary: List[types.Metrics]


class BraxOptimizer(BaseOptimizer[BraxState, BraxOutput]):
    def __init__(self,
                 agent_class,
                 system: System,
                 true_buffer: UniformSamplingQueue,
                 **agent_kwargs):
        super().__init__(system)
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.true_buffer = true_buffer
        if system is None:
            self.dummy_trainer = None
            self.make_policy = None
        else:
            self.set_system(system)

    def set_system(self, system: System):
        self.key, sys_key, buffer_key = jr.split(self.key, 3)
        dummy_true_buffer_state = self.dummy_true_buffer_state(buffer_key)
        super().set_system(system)
        dummy_env = BraxWrapper(system=self.system,
                                system_params=self.system.init_params(sys_key),
                                sample_buffer_state=dummy_true_buffer_state,
                                sample_buffer=self.true_buffer)
        self.dummy_trainer = self.agent_class(environment=dummy_env, **self.agent_kwargs)
        self.make_policy = self.dummy_trainer.make_policy

    def init(self,
             key: chex.PRNGKey,
             true_buffer_state: ReplayBufferState | None = None) -> BraxState:
        assert self.system is not None, "Brax optimizer requires system to be defined."
        if true_buffer_state is None:
            dummy_buffer_key, key = jr.split(key, 2)
            true_buffer_state = self.dummy_true_buffer_state(dummy_buffer_key)
        keys = jr.split(key, 3)
        system_params = self.system.init_params(keys[0])
        training_state = self.dummy_trainer.init_training_state(keys[1])
        return BraxState(system_params=system_params,
                         true_buffer_state=true_buffer_state,
                         policy_params=training_state.get_policy_params(),
                         key=keys[2])

    @partial(jit, static_argnums=(0, 3))
    def act(self,
            obs: chex.Array,
            opt_state: BraxState[DynamicsParams, RewardParams],
            evaluate: bool = True) -> Tuple[chex.Array, BraxState[DynamicsParams, RewardParams]]:
        assert self.system is not None, "Brax optimizer requires system to be defined."
        policy = self.make_policy(opt_state.policy_params, evaluate)
        key, subkey = jr.split(opt_state.key)
        action = policy(obs, subkey)[0]
        return action, opt_state.replace(key=key)

    # @partial(jit, static_argnums=(0,))
    def train(self,
              opt_state: BraxState
              ) -> BraxOutput:
        assert self.system is not None, "Brax optimizer requires system to be defined."
        env = BraxWrapper(system=self.system,
                          system_params=opt_state.system_params,
                          sample_buffer_state=opt_state.true_buffer_state,
                          sample_buffer=self.true_buffer)

        trainer = self.agent_class(environment=env, **self.agent_kwargs)
        key, new_key = jr.split(opt_state.key)
        policy_params, metrics = trainer.run_training(key=new_key)
        new_opt_state = opt_state.replace(policy_params=policy_params, key=new_key)
        return BraxOutput(optimizer_state=new_opt_state, summary=metrics)


class PPOOptimizer(BraxOptimizer):
    def __init__(self,
                 system: System,
                 true_buffer: UniformSamplingQueue,
                 **ppo_kwargs):
        super().__init__(agent_class=PPO, system=system, true_buffer=true_buffer, **ppo_kwargs)


class SACOptimizer(BraxOptimizer):
    def __init__(self,
                 system: System,
                 true_buffer: UniformSamplingQueue,
                 **sac_kwargs):
        super().__init__(agent_class=SAC, system=system, true_buffer=true_buffer, **sac_kwargs)
