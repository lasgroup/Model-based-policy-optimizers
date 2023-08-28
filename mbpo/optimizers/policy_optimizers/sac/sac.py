import functools
import math
import time
from functools import partial
from typing import Any, Callable, Optional, Tuple, Sequence

import chex
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import wandb
from brax import envs
from brax.training import networks
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from flax import struct
from jax import jit
from jax.lax import scan
from jaxtyping import PyTree

from mbpo.optimizers.policy_optimizers.brax_utils.training import wrap as wrap_for_training
from mbpo.optimizers.policy_optimizers.sac import acting
from mbpo.optimizers.policy_optimizers.sac.losses import SACLosses
from mbpo.optimizers.policy_optimizers.sac.sac_networks import SACNetworksModel, make_inference_fn
from mbpo.optimizers.policy_optimizers.sac.utils import gradient_update_fn, metrics_to_float

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any


@struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState

    def get_policy_params(self):
        return self.normalizer_params, self.policy_params


class SAC:
    def __init__(self,
                 environment: envs.Env,
                 num_timesteps: int,
                 episode_length: int,
                 action_repeat: int = 1,
                 num_env_steps_between_updates: int = 2,
                 num_envs: int = 1,
                 num_eval_envs: int = 128,
                 lr_alpha: float = 1e-4,
                 lr_policy: float = 1e-4,
                 lr_q: float = 1e-4,
                 wd_alpha: float = 0.,
                 wd_policy: float = 0.,
                 wd_q: float = 0.,
                 discounting: float = 0.9,
                 batch_size: int = 256,
                 num_evals: int = 1,
                 normalize_observations: bool = False,
                 reward_scaling: float = 1.,
                 tau: float = 0.005,
                 min_replay_size: int = 0,
                 max_replay_size: Optional[int] = None,
                 grad_updates_per_step: int = 1,
                 deterministic_eval: bool = True,
                 init_log_alpha: float = 0.,
                 target_entropy: float | None = None,
                 policy_hidden_layer_sizes: Sequence[int] = (64, 64, 64),
                 policy_activation: networks.ActivationFn = nn.swish,
                 critic_hidden_layer_sizes: Sequence[int] = (64, 64, 64),
                 critic_activation: networks.ActivationFn = nn.swish,
                 wandb_logging: bool = False,
                 return_best_model: bool = False,
                 eval_environment: envs.Env | None = None,
                 episode_length_eval: int | None = None,
                 eval_key_fixed: bool = False,
                 ):
        if min_replay_size >= num_timesteps:
            raise ValueError(
                'No training will happen because min_replay_size >= num_timesteps')

        self.eval_key_fixed = eval_key_fixed
        self.return_best_model = return_best_model
        self.target_entropy = target_entropy
        self.init_log_alpha = init_log_alpha
        self.wandb_logging = wandb_logging
        self.min_replay_size = min_replay_size
        self.num_timesteps = num_timesteps
        self.num_envs = num_envs
        self.deterministic_eval = deterministic_eval
        self.num_eval_envs = num_eval_envs
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.num_evals = num_evals
        self.num_env_steps_between_updates = num_env_steps_between_updates

        if max_replay_size is None:
            max_replay_size = num_timesteps
        self.max_replay_size = max_replay_size
        # The number of environment steps executed for every `actor_step()` call.
        self.env_steps_per_actor_step = action_repeat * num_envs
        # equals to ceil(min_replay_size / num_envs)
        self.num_prefill_actor_steps = math.ceil(min_replay_size / num_envs)
        num_prefill_env_steps = self.num_prefill_actor_steps * self.env_steps_per_actor_step
        assert num_timesteps - num_prefill_env_steps >= 0
        self.num_evals_after_init = max(num_evals - 1, 1)
        # The number of run_one_sac_epoch calls per run_sac_training.
        # equals to
        num_env_steps_in_one_train_step = self.num_evals_after_init * self.env_steps_per_actor_step
        num_env_steps_in_one_train_step *= num_env_steps_between_updates
        self.num_training_steps_per_epoch = math.ceil(
            (num_timesteps - num_prefill_env_steps) / num_env_steps_in_one_train_step)
        # num_training_steps_per_epoch is how many action we apply in every epoch

        self.grad_updates_per_step = grad_updates_per_step

        self.tau = tau
        self.env = wrap_for_training(environment,
                                     episode_length=episode_length,
                                     action_repeat=action_repeat)

        # Prepare env for evaluation
        if episode_length_eval is None:
            episode_length_eval = episode_length

        self.episode_length_eval = episode_length_eval
        if eval_environment is None:
            eval_environment = environment
        self.eval_env = wrap_for_training(eval_environment,
                                          episode_length=episode_length_eval,
                                          action_repeat=action_repeat)

        self.x_dim = self.env.observation_size
        self.u_dim = self.env.action_size

        def normalize_fn(batch: PyTree, _: PyTree) -> PyTree:
            return batch

        if normalize_observations:
            normalize_fn = running_statistics.normalize
        self.normalize_fn = normalize_fn

        self.sac_networks_model = SACNetworksModel(
            x_dim=self.x_dim, u_dim=self.u_dim,
            preprocess_observations_fn=normalize_fn,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            policy_activation=policy_activation,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_activation=critic_activation)

        self.make_policy = make_inference_fn(self.sac_networks_model.get_sac_networks())

        self.alpha_optimizer = optax.adamw(learning_rate=lr_alpha, weight_decay=wd_alpha)
        self.policy_optimizer = optax.adamw(learning_rate=lr_policy, weight_decay=wd_policy)
        self.q_optimizer = optax.adamw(learning_rate=lr_q, weight_decay=wd_q)

        # Set this to None unless you want to use parallelism across multiple devices.
        self._PMAP_AXIS_NAME = None

        # Setup replay buffer.
        dummy_obs = jnp.zeros((self.x_dim,))
        dummy_action = jnp.zeros((self.u_dim,))
        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=jnp.array(0.),
            discount=jnp.array(0.),
            next_observation=dummy_obs,
            extras={'state_extras': {'truncation': 0.}, 'policy_extras': {}})

        self.replay_buffer = replay_buffers.UniformSamplingQueue(
            max_replay_size=max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=batch_size * grad_updates_per_step)

        # Setup optimization
        self.losses = SACLosses(sac_network=self.sac_networks_model.get_sac_networks(), reward_scaling=reward_scaling,
                                discounting=discounting, u_dim=self.u_dim, target_entropy=self.target_entropy)

        self.alpha_update = gradient_update_fn(
            self.losses.alpha_loss, self.alpha_optimizer, pmap_axis_name=self._PMAP_AXIS_NAME)
        self.critic_update = gradient_update_fn(
            self.losses.critic_loss, self.q_optimizer, pmap_axis_name=self._PMAP_AXIS_NAME)
        self.actor_update = gradient_update_fn(
            self.losses.actor_loss, self.policy_optimizer, pmap_axis_name=self._PMAP_AXIS_NAME)

    @partial(jit, static_argnums=(0,))
    def sgd_step(self, carry: Tuple[TrainingState, chex.PRNGKey],
                 transitions: Transition) -> Tuple[Tuple[TrainingState, chex.PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor = jr.split(key, 4)

        alpha_loss, alpha_params, alpha_optimizer_state = self.alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state)
        alpha = jnp.exp(training_state.alpha_params)
        critic_loss, q_params, q_optimizer_state = self.critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state)
        actor_loss, policy_params, policy_optimizer_state = self.actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state)

        new_target_q_params = jtu.tree_map(lambda x, y: x * (1 - self.tau) + y * self.tau,
                                           training_state.target_q_params, q_params)

        metrics = {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params),
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params)
        return (new_training_state, key), metrics

    def get_experience(self, normalizer_params: running_statistics.RunningStatisticsState, policy_params: Params,
                       env_state: envs.State, buffer_state: ReplayBufferState, key: chex.PRNGKey
                       ) -> Tuple[running_statistics.RunningStatisticsState, envs.State, ReplayBufferState]:
        policy = self.make_policy((normalizer_params, policy_params))

        def f(carry, _):
            k, es = carry
            k, k_t = jr.split(k)
            new_es, new_trans = acting.actor_step(self.env, es, policy, k_t, extra_fields=('truncation',))
            return (k, new_es), new_trans

        (key, env_state), transitions = scan(f, (key, env_state), jnp.arange(self.num_env_steps_between_updates))

        transitions = jtu.tree_map(jnp.concatenate, transitions)

        normalizer_params = running_statistics.update(
            normalizer_params,
            transitions.observation,
            pmap_axis_name=self._PMAP_AXIS_NAME)

        buffer_state = self.replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    def training_step(self, training_state: TrainingState, env_state: envs.State, buffer_state: ReplayBufferState,
                      key: chex.PRNGKey) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        experience_key, training_key = jr.split(key)

        normalizer_params, env_state, buffer_state = self.get_experience(
            training_state.normalizer_params, training_state.policy_params,
            env_state, buffer_state, experience_key)

        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + self.env_steps_per_actor_step * self.num_env_steps_between_updates)

        buffer_state, transitions = self.replay_buffer.sample(buffer_state)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jtu.tree_map(
            lambda x: jnp.reshape(x, (self.grad_updates_per_step, -1) + x.shape[1:]),
            transitions)
        (training_state, _), metrics = scan(self.sgd_step, (training_state, training_key), transitions)

        metrics['buffer_current_size'] = jnp.array(self.replay_buffer.size(buffer_state))
        return training_state, env_state, buffer_state, metrics

    @partial(jit, static_argnums=(0,))
    def prefill_replay_buffer(self, training_state: TrainingState, env_state: envs.State,
                              buffer_state: ReplayBufferState, key: chex.PRNGKey
                              ) -> Tuple[TrainingState, envs.State, ReplayBufferState, chex.PRNGKey]:

        def f(carry, _):
            tr_step, e_state, b_state, ky = carry
            ky, new_key = jr.split(ky)
            new_normalizer_params, e_state, b_state = self.get_experience(
                tr_step.normalizer_params, tr_step.policy_params,
                e_state, b_state, ky)
            new_training_state = tr_step.replace(
                normalizer_params=new_normalizer_params,
                env_steps=tr_step.env_steps + self.env_steps_per_actor_step)
            return (new_training_state, e_state, b_state, new_key), ()

        return scan(f, (training_state, env_state, buffer_state, key), (), length=self.num_prefill_actor_steps)[0]

    @partial(jit, static_argnums=(0,))
    def training_epoch(self, training_state: TrainingState, env_state: envs.State, buffer_state: ReplayBufferState,
                       key: chex.PRNGKey) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

        def f(carry, _):
            ts, es, bs, k = carry
            k, new_key = jr.split(k)
            ts, es, bs, metr = self.training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metr

        (training_state, env_state, buffer_state, key), metrics = scan(
            f, (training_state, env_state, buffer_state, key), (),
            length=self.num_training_steps_per_epoch)
        metrics = jtu.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    def training_epoch_with_timing(self, training_state: TrainingState, env_state: envs.State,
                                   buffer_state: ReplayBufferState, key: chex.PRNGKey
                                   ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        t = time.time()
        training_state, env_state, buffer_state, metrics = self.training_epoch(training_state, env_state, buffer_state,
                                                                               key)

        epoch_training_time = time.time() - t
        sps = (self.env_steps_per_actor_step * self.num_training_steps_per_epoch) / epoch_training_time
        metrics = {'training/sps': jnp.array(sps),
                   **{f'training/{name}': jnp.array(value) for name, value in metrics.items()}}
        return training_state, env_state, buffer_state, metrics

    def init_training_state(self, key: chex.PRNGKey) -> TrainingState:
        """Inits the training state and replicates it over devices."""
        key_policy, key_q = jr.split(key)
        log_alpha = jnp.asarray(self.init_log_alpha, dtype=jnp.float32)
        alpha_optimizer_state = self.alpha_optimizer.init(log_alpha)

        policy_params = self.sac_networks_model.get_policy_network().init(key_policy)
        policy_optimizer_state = self.policy_optimizer.init(policy_params)

        q_params = self.sac_networks_model.get_q_network().init(key_q)
        q_optimizer_state = self.q_optimizer.init(q_params)

        normalizer_params = running_statistics.init_state(
            specs.Array((self.x_dim,), jnp.float32))

        training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=q_params,
            gradient_steps=jnp.zeros(()),
            env_steps=jnp.zeros(()),
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=log_alpha,
            normalizer_params=normalizer_params)
        return training_state

    def run_training(self, key: chex.PRNGKey, progress_fn: Callable[[int, Metrics], None] = lambda *args: None):
        key, subkey = jr.split(key)
        training_state = self.init_training_state(subkey)

        key, rb_key, env_key, eval_key = jr.split(key, 4)

        # Initialize initial env state
        env_keys = jr.split(env_key, self.num_envs)
        env_state = self.env.reset(env_keys)

        # Initialize replay buffer
        buffer_state = self.replay_buffer.init(rb_key)

        evaluator = acting.Evaluator(
            self.eval_env, functools.partial(self.make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=self.num_eval_envs, episode_length=self.episode_length_eval, action_repeat=self.action_repeat,
            key=eval_key)

        # Run initial eval
        all_metrics = []
        metrics = {}
        highest_eval_episode_reward = jnp.array(-jnp.inf)
        best_params = (training_state.normalizer_params, training_state.policy_params)
        if self.num_evals > 1:
            metrics = evaluator.run_evaluation((training_state.normalizer_params, training_state.policy_params),
                                               training_metrics={})
            if metrics['eval/episode_reward'] > highest_eval_episode_reward:
                highest_eval_episode_reward = metrics['eval/episode_reward']
                best_params = (training_state.normalizer_params, training_state.policy_params)
            if self.wandb_logging:
                metrics = metrics_to_float(metrics)
                wandb.log(metrics)
            all_metrics.append(metrics)
            progress_fn(0, metrics)

        # Create and initialize the replay buffer.
        key, prefill_key = jr.split(key)
        training_state, env_state, buffer_state, _ = self.prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key)

        replay_size = self.replay_buffer.size(buffer_state)
        if self.wandb_logging:
            wandb.log(metrics_to_float({'replay size after prefill': replay_size}))
        # assert replay_size >= self.min_replay_size

        current_step = 0

        if self.eval_key_fixed:
            key, eval_key = jr.split(key)

        for _ in range(self.num_evals_after_init):
            if self.wandb_logging:
                wandb.log(metrics_to_float({'step': current_step}))

            # Optimization
            key, epoch_key = jr.split(key)
            training_state, env_state, buffer_state, training_metrics = self.training_epoch_with_timing(training_state,
                                                                                                        env_state,
                                                                                                        buffer_state,
                                                                                                        epoch_key)
            # current_step = int(training_state.env_steps)

            # Eval and logging
            # Run evals.
            if not self.eval_key_fixed:
                key, eval_key = jr.split(key)
            metrics = evaluator.run_evaluation((training_state.normalizer_params, training_state.policy_params),
                                               training_metrics, unroll_key=eval_key)

            if metrics['eval/episode_reward'] > highest_eval_episode_reward:
                highest_eval_episode_reward = metrics['eval/episode_reward']
                best_params = (training_state.normalizer_params, training_state.policy_params)

            if self.wandb_logging:
                metrics = metrics_to_float(metrics)
                wandb.log(metrics)
            all_metrics.append(metrics)
            progress_fn(training_state.env_steps, metrics)

        # total_steps = current_step
        # assert total_steps >= self.num_timesteps
        last_params = (training_state.normalizer_params, training_state.policy_params)

        if self.return_best_model:
            params_to_return = best_params
        else:
            params_to_return = last_params

        if self.wandb_logging:
            wandb.log(metrics_to_float({'total steps': int(training_state.env_steps)}))
        return params_to_return, all_metrics
