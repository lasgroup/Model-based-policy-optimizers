from typing import Sequence, Tuple

import chex
from brax.training import distribution
from brax.training import networks
from brax.training import types
from flax import linen as nn
from flax import struct


@struct.dataclass
class PPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


class PPONetworksModel:
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 preprocess_observations_fn: types.PreprocessObservationFn = types
                 .identity_observation_preprocessor,
                 policy_hidden_layer_sizes: Sequence[int] = (64, 64),
                 policy_activation: networks.ActivationFn = nn.swish,
                 value_hidden_layer_sizes: Sequence[int] = (64, 64, 64),
                 value_activation: networks.ActivationFn = nn.swish,
                 ):
        """Make PPO networks with preprocessor."""

        self._parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=u_dim)
        self._policy_network = networks.make_policy_network(
            self._parametric_action_distribution.param_size, x_dim,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=policy_hidden_layer_sizes,
            activation=policy_activation)
        self._value_network = networks.make_value_network(
            x_dim,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=value_activation)
        self._ppo_networks = PPONetworks(self._policy_network, self._value_network,
                                         self._parametric_action_distribution)

    def get_ppo_networks(self):
        return self._ppo_networks

    def get_policy_network(self):
        return self._policy_network

    def get_value_network(self):
        return self._value_network

    def get_parametric_action_distribution(self):
        return self._parametric_action_distribution


def make_inference_fn(ppo_networks: PPONetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: types.PolicyParams,
                    deterministic: bool = False) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(observations: types.Observation,
                   key_sample: chex.PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = policy_network.apply(*params, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample)
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions)
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions
            }

        return policy

    return make_policy
