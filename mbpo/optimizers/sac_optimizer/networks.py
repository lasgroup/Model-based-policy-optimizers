import dataclasses
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from brax.training import types
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(linen.Module):
    """MLP module."""
    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(
                hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden


def make_q_network(
        x_dim: int,
        u_dim: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu,
        n_critics: int = 2) -> FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""
        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = MLP(
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform())(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, x_dim))
    dummy_action = jnp.zeros((1, u_dim))
    return FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)


def make_policy_network(
        param_size: int,
        x_dim: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types
        .identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform())

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, x_dim))
    # Feed forward neural network is already in the in vectorized form (i.e. it expects multiple observations as input)
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply)
