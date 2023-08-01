from typing import Sequence, Tuple

import chex
from brax.training import distribution
from brax.training import networks
from brax.training import types
from flax import linen as nn
from flax import struct


@struct.dataclass
class SACNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


class SACNetworksModel:
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 preprocess_observations_fn: types.PreprocessObservationFn = types
                 .identity_observation_preprocessor,
                 policy_hidden_layer_sizes: Sequence[int] = (64, 64),
                 policy_activation: networks.ActivationFn = nn.swish,
                 critic_hidden_layer_sizes: Sequence[int] = (64, 64, 64),
                 critic_activation: networks.ActivationFn = nn.swish,
                 ):
        """Make SAC networks."""

        self._parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=u_dim)
        self._policy_network = networks.make_policy_network(
            self._parametric_action_distribution.param_size, x_dim,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=policy_activation)
        self._q_network = networks.make_q_network(
            x_dim, u_dim,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=critic_hidden_layer_sizes,
            activation=critic_activation)
        self._sac_networks = SACNetworks(self._policy_network, self._q_network, self._parametric_action_distribution)

    def get_sac_networks(self):
        return self._sac_networks

    def get_policy_network(self):
        return self._policy_network

    def get_q_network(self):
        return self._q_network

    def get_parametric_action_distribution(self):
        return self._parametric_action_distribution


def make_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_policy(params: types.PolicyParams,
                    deterministic: bool = False) -> types.Policy:
        def policy(observations: types.Observation,
                   key_sample: chex.PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = sac_networks.policy_network.apply(*params, observations)
            if deterministic:
                return sac_networks.parametric_action_distribution.mode(logits), {}
            return sac_networks.parametric_action_distribution.sample(
                logits, key_sample), {}

        return policy

    return make_policy
