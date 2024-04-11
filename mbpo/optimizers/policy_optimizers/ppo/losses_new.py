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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params


@chex.dataclass
class PPONetworkParams:
    """Contains training state for the learner."""
    policy: Params
    value: Params


class PPOLoss:
    def __init__(self,
                 ppo_network: ppo_networks.PPONetworks,
                 entropy_cost: float,
                 discounting: float,
                 reward_scaling: float,
                 gae_lambda: float,
                 clipping_epsilon: float,
                 normalize_advantage: bool,
                 non_equidistant_time: bool = False,
                 continuous_discounting: float = 0,
                 min_time_between_switches: float = 0,
                 max_time_between_switches: float = 0,
                 env_dt: float = 0,
                 ):
        self.ppo_network = ppo_network
        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.gae_lambda = gae_lambda
        self.clipping_epsilon = clipping_epsilon
        self.normalize_advantage = normalize_advantage
        self.non_equidistant_time = non_equidistant_time
        self.continuous_discounting = continuous_discounting
        self.min_time_between_switches = min_time_between_switches
        self.max_time_between_switches = max_time_between_switches
        self.env_dt = env_dt

    def loss(self,
             params: PPONetworkParams,
             normalizer_params: Any,
             data: types.Transition,
             rng: jnp.ndarray, ):
        """Computes PPO loss.

        Args:
          params: Network parameters,
          normalizer_params: Parameters of the normalizer.
          data: Transition that with leading dimension [B, T]. extra fields required
            are ['state_extras']['truncation'] ['policy_extras']['raw_action']
              ['policy_extras']['log_prob']
          rng: Random key

        Returns:
          A tuple (loss, metrics)
        """
        parametric_action_distribution = self.ppo_network.parametric_action_distribution
        policy_apply = self.ppo_network.policy_network.apply
        value_apply = self.ppo_network.value_network.apply

        # Put the time dimension first.
        data = jtu.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        policy_logits = policy_apply(normalizer_params, params.policy, data.observation)

        baseline = value_apply(normalizer_params, params.value, data.observation)

        bootstrap_value = value_apply(normalizer_params, params.value,
                                      data.next_observation[-1])

        rewards = data.reward * self.reward_scaling
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        target_action_log_probs = parametric_action_distribution.log_prob(
            policy_logits, data.extras['policy_extras']['raw_action'])
        behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

        if self.non_equidistant_time:
            pseudo_time_for_action = data.action[..., -1]
            t_lower = self.min_time_between_switches
            t_upper = self.max_time_between_switches
            time_for_action = ((t_upper - t_lower) / 2 * pseudo_time_for_action + (t_upper + t_lower) / 2)
            time_for_action = (time_for_action // self.env_dt) * self.env_dt
            discounting = jnp.exp(- self.continuous_discounting * time_for_action)

        vs, advantages = self.compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            discounting=discounting if self.non_equidistant_time else None
        )
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = jnp.clip(rho_s, 1 - self.clipping_epsilon,
                                   1 + self.clipping_epsilon) * advantages

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        # Why is here double times 0.5 factor?
        v_loss = jnp.mean(v_error * v_error) * 0.5

        # Entropy reward
        entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
        entropy_loss = self.entropy_cost * -entropy

        total_loss = policy_loss + v_loss + entropy_loss
        return total_loss, {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'entropy_loss': entropy_loss
        }

    def compute_gae(self,
                    truncation: jnp.ndarray,
                    termination: jnp.ndarray,
                    rewards: jnp.ndarray,
                    values: jnp.ndarray,
                    bootstrap_value: jnp.ndarray,
                    discounting: jnp.ndarray | None = None
                    ):
        """Calculates the Generalized Advantage Estimation (GAE).

        Args:
          truncation: A float32 tensor of shape [T, B] with truncation signal.
          termination: A float32 tensor of shape [T, B] with termination signal.
          rewards: A float32 tensor of shape [T, B] containing rewards generated by
            following the behaviour policy.
          values: A float32 tensor of shape [T, B] with the value function estimates
            wrt. the target policy.
          bootstrap_value: A float32 of shape [B] with the value function estimate at
            time T.
          lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
            lambda_=1.
          discount: TD discount.

        Returns:
          A float32 tensor of shape [T, B]. Can be used as target to
            train a baseline (V(x_t) - vs_t)^2.
          A float32 tensor of shape [T, B] of advantages.
        """

        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = jnp.concatenate(
            [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
        if self.non_equidistant_time:
            deltas = rewards + discounting * (1 - termination) * values_t_plus_1 - values
        else:
            deltas = rewards + self.discounting * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = jnp.zeros_like(bootstrap_value)
        vs_minus_v_xs = []

        if self.non_equidistant_time:
            def compute_vs_minus_v_xs(carry, target_t):
                acc = carry
                truncation_mask, delta, termination, discounting = target_t
                acc = delta + discounting * (1 - termination) * truncation_mask * self.gae_lambda * acc
                return acc, acc

            _, (vs_minus_v_xs) = jax.lax.scan(
                compute_vs_minus_v_xs, acc,
                (truncation_mask, deltas, termination, discounting),
                length=int(truncation_mask.shape[0]),
                reverse=True)

        else:
            def compute_vs_minus_v_xs(carry, target_t):
                acc = carry
                truncation_mask, delta, termination = target_t
                acc = delta + self.discounting * (1 - termination) * truncation_mask * self.gae_lambda * acc
                return acc, acc

            _, (vs_minus_v_xs) = jax.lax.scan(
                compute_vs_minus_v_xs, acc,
                (truncation_mask, deltas, termination),
                length=int(truncation_mask.shape[0]),
                reverse=True)

        # Add V(x_s) to get v_s.
        vs = jnp.add(vs_minus_v_xs, values)

        vs_t_plus_1 = jnp.concatenate(
            [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
        if self.non_equidistant_time:
            advantages = (rewards + discounting *
                          (1 - termination) * vs_t_plus_1 - values) * truncation_mask
        else:
            advantages = (rewards + self.discounting *
                          (1 - termination) * vs_t_plus_1 - values) * truncation_mask
        return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)
