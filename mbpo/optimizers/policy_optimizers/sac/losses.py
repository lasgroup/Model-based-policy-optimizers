# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any

import chex
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params

from mbpo.optimizers.policy_optimizers.sac.sac_networks import SACNetworks

Transition = types.Transition


class SACLosses:
    def __init__(self,
                 sac_network: SACNetworks,
                 reward_scaling: float,
                 discounting: float,
                 u_dim: int,
                 target_entropy: float | None = None,
                 non_equidistant_time: bool = False,
                 continuous_discounting: float = 0,
                 min_time_between_switches: float = 0,
                 max_time_between_switches: float = 0,
                 ):
        self.sac_network = sac_network
        self.reward_scaling = reward_scaling
        self.discounting = discounting
        self.u_dim = u_dim
        if target_entropy is None:
            target_entropy = -0.5 * self.u_dim
        self.target_entropy = target_entropy
        self.policy_network = self.sac_network.policy_network
        self.q_network = self.sac_network.q_network
        self.parametric_action_distribution = self.sac_network.parametric_action_distribution
        self.non_equidistant_time = non_equidistant_time
        self.continuous_discounting = continuous_discounting
        self.min_time_between_switches = min_time_between_switches
        self.max_time_between_switches = max_time_between_switches

    def alpha_loss(self, log_alpha: jnp.ndarray, policy_params: Params,
                   normalizer_params: Any, transitions: Transition,
                   key: chex.PRNGKey) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = self.policy_network.apply(normalizer_params, policy_params,
                                                transitions.observation)
        action = self.parametric_action_distribution.sample_no_postprocessing(
            dist_params, key)
        log_prob = self.parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - self.target_entropy)
        return jnp.mean(alpha_loss)

    def critic_loss(self, q_params: Params, policy_params: Params,
                    normalizer_params: Any, target_q_params: Params,
                    alpha: jnp.ndarray, transitions: Transition,
                    key: chex.PRNGKey) -> jnp.ndarray:
        q_old_action = self.q_network.apply(normalizer_params, q_params,
                                            transitions.observation, transitions.action)
        next_dist_params = self.policy_network.apply(normalizer_params, policy_params,
                                                     transitions.next_observation)
        next_action = self.parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key)
        next_log_prob = self.parametric_action_distribution.log_prob(
            next_dist_params, next_action)
        next_action = self.parametric_action_distribution.postprocess(next_action)
        next_q = self.q_network.apply(normalizer_params, target_q_params,
                                      transitions.next_observation, next_action)
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        if self.non_equidistant_time:
            pseudo_time_for_action = transitions.action[..., -1]
            t_lower = self.min_time_between_switches
            t_upper = self.max_time_between_switches
            time_for_action = ((t_upper - t_lower) / 2 * pseudo_time_for_action + (t_upper + t_lower) / 2)
            discounting = jnp.exp(- self.continuous_discounting * time_for_action)
        else:
            discounting = self.discounting

        target_q = jax.lax.stop_gradient(transitions.reward * self.reward_scaling +
                                         transitions.discount * discounting *
                                         next_v)
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        truncation = transitions.extras['state_extras']['truncation']
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    def actor_loss(self, policy_params: Params, normalizer_params: Any,
                   q_params: Params, alpha: jnp.ndarray, transitions: Transition,
                   key: chex.PRNGKey) -> jnp.ndarray:
        dist_params = self.policy_network.apply(normalizer_params, policy_params,
                                                transitions.observation)
        action = self.parametric_action_distribution.sample_no_postprocessing(
            dist_params, key)
        log_prob = self.parametric_action_distribution.log_prob(dist_params, action)
        action = self.parametric_action_distribution.postprocess(action)
        q_action = self.q_network.apply(normalizer_params, q_params,
                                        transitions.observation, action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q
        return jnp.mean(actor_loss)
