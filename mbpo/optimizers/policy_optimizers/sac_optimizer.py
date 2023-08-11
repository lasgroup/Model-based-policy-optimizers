from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState

from mbpo.optimizers.policy_optimizers.brax_optimizer import BraxOptimizer
from mbpo.optimizers.policy_optimizers.sac.sac import SAC
from mbpo.systems.base_systems import System


class SACOptimizer(BraxOptimizer):
    def __init__(self,
                 system: System,
                 true_buffer: UniformSamplingQueue,
                 dummy_true_buffer_state: ReplayBufferState,
                 **sac_kwargs):
        super().__init__(agent_class=SAC, system=system, true_buffer=true_buffer,
                         dummy_true_buffer_state=dummy_true_buffer_state, **sac_kwargs)
