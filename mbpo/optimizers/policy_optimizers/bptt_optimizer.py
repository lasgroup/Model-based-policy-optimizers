from typing import Union, Callable, Sequence, Tuple, Any, Optional
from jaxtyping import PyTree
import distrax
import jax
import jax.numpy as jnp
import flax.linen as nn
from bsm.utils.network_utils import MLP
import chex
from mbpo.optimizers.base_optimizer import BaseOptimizer
from mbpo.systems import SystemParams
from brax.training.types import Transition
from jax import flatten_util
from brax.training.replay_buffers import ReplayBufferState, UniformSamplingQueue
import optax
from copy import deepcopy
import flax.struct as struct
import functools
from mbpo.utils.optimizer_utils import rollout_policy, lambda_return, soft_update
import math
from optax import l2_loss
from brax.training import networks

EPS = 1e-8


@struct.dataclass
class NormalizerState:
    mean: jax.Array
    std: jax.Array
    size: int


class Normalizer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def initialize_normalizer_state(self):
        mean = jnp.zeros(*self.input_shape)
        std = jnp.ones(*self.input_shape)
        size = 0
        return NormalizerState(
            mean=mean,
            std=std,
            size=size,
        )

    @staticmethod
    @jax.jit
    def update(x: jax.Array, state: NormalizerState):
        new_size = x.shape[0]
        total_size = new_size + state.size
        new_mean = (state.mean * state.size + jnp.sum(x, axis=0)) / total_size
        new_s_n = jnp.square(state.std) * state.size + jnp.sum(jnp.square(x - new_mean),
                                                               axis=0
                                                               ) + state.size * jnp.square(state.mean -
                                                                                           new_mean)
        new_var = new_s_n / total_size
        new_std = jnp.sqrt(new_var)
        mean = new_mean
        std = jnp.maximum(new_std, jnp.ones_like(new_std) * EPS)
        size = total_size
        return NormalizerState(mean=mean, std=std, size=size)

    @staticmethod
    @jax.jit
    def normalize(x, state: NormalizerState):
        return (x - state.mean) / state.std

    @staticmethod
    @jax.jit
    def inverse(x, state: NormalizerState):
        return x * state.std + state.mean


@chex.dataclass
class BPTTState:
    actor_opt_state: optax.OptState
    actor_params: Any
    critic_opt_state: optax.OptState
    critic_params: Any
    target_critic_params: Any
    state_normalizer_state: NormalizerState
    reward_normalizer_state: NormalizerState
    key: jax.random.PRNGKeyArray


@chex.dataclass
class BPTTAgentSummary:
    actor_grad_norm: jax.Array = struct.field(default_factory=lambda: jnp.zeros(1).squeeze(-1))
    critic_grad_norm: jax.Array = struct.field(default_factory=lambda: jnp.zeros(1).squeeze(-1))
    actor_loss: jax.Array = struct.field(default_factory=lambda: jnp.zeros(1).squeeze(-1))
    critic_loss: jax.Array = struct.field(default_factory=lambda: jnp.zeros(1).squeeze(-1))
    reward: jax.Array = struct.field(default_factory=lambda: jnp.zeros(1).squeeze(-1))
    best_reward: jax.Array = struct.field(default_factory=lambda: jnp.ones(1).squeeze(-1) * -jnp.inf)


@chex.dataclass
class BPTTTrainingOutput:
    bptt_state: BPTTState
    bptt_summary: BPTTAgentSummary


def inv_softplus(x: Union[chex.Array, float]) -> chex.Array:
    return jnp.where(x < 20.0, jnp.log(jnp.exp(x) - 1.0), x)


def atanh(x: jax.Array) -> jax.Array:
    """
    Inverse of Tanh

    Taken from Pyro: https://github.com/pyro-ppl/pyro
    0.5 * torch.log((1 + x ) / (1 - x))
    """
    x = jnp.clip(x, -1 + EPS, 1 - EPS)
    y = 0.5 * jnp.log((1 + x) / (1 - x))
    return y


class Actor(nn.Module):
    features: Sequence[int]
    action_dim: int
    activation: Callable = nn.swish
    init_stddev: float = float(1.0)
    sig_min: float = float(1e-6)
    sig_max: float = float(1e2)

    @nn.compact
    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        actor_net = MLP(features=self.features,
                        output_dim=2 * self.action_dim,
                        activation=self.activation)

        out = actor_net(obs)
        mu, sig = jnp.split(out, 2, axis=-1)
        init_std = inv_softplus(self.init_stddev)
        sig = nn.softplus(sig + init_std)
        sig = jnp.clip(sig, self.sig_min, self.sig_max)
        return mu, sig

    def get_log_prob(self, squashed_action: jax.Array, obs: jax.Array, params: PyTree):
        mu, sig = self.apply(params, obs)
        u = atanh(squashed_action)
        dist = distrax.Normal(loc=mu, scale=sig)
        log_l = dist.log_prob(u)
        log_l -= jnp.sum(
            jnp.log((1 - jnp.square(squashed_action))), axis=-1
        )
        return log_l.reshape(-1, 1)


class Critic(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        critic_1 = MLP(
            features=self.features,
            output_dim=1,
            activation=self.activation)

        critic_2 = MLP(
            features=self.features,
            output_dim=1,
            activation=self.activation)
        value_1 = critic_1(obs).squeeze(-1)
        value_2 = critic_2(obs).squeeze(-1)
        return value_1, value_2


class BPTTOptimizer(BaseOptimizer):

    def __init__(self,
                 action_dim: int,
                 obs_dim: int,
                 horizon: int = 20,
                 num_samples_per_gradient_update: int = 10,
                 train_steps: int = 20,
                 normalize: bool = True,
                 action_normalize: bool = True,
                 actor_features: Sequence[int] = (64, 64, 64),
                 policy_activation: networks.ActivationFn = nn.swish,
                 critic_features: Sequence[int] = (64, 64, 64),
                 critic_activation: networks.ActivationFn = nn.swish,
                 init_stddev: float = 1.0,
                 lr_actor: float = 1e-3,
                 weight_decay_actor: float = 1e-5,
                 lr_critic: float = 1e-3,
                 weight_decay_critic: float = 1e-5,
                 reset_optimizer: bool = True,
                 target_soft_update_tau: float = 0.005,
                 rng: jax.Array = jax.random.PRNGKey(0),
                 evaluation_samples: int = 100,
                 evaluation_horizon: int = 100,
                 evaluation_frequency: int = -1,
                 critic_updates_per_policy_update: int = 1,
                 discount: float = 0.99,
                 lambda_: float = 0.97,
                 loss_ent_coefficient: float = 0.005,
                 use_best_trained_policy: bool = False,
                 sample_simulated_transitions: bool = True,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.state_normalizer = Normalizer((obs_dim,))
        self.reward_normalizer = Normalizer((1,))
        sample_obs = jnp.ones(obs_dim, )
        self.actor = Actor(features=actor_features, action_dim=action_dim, init_stddev=init_stddev,
                           activation=policy_activation)
        self.critic = Critic(features=critic_features, activation=critic_activation)
        actor_rng, critic_rng, rng = jax.random.split(rng, 3)
        critic_params = self.critic.init(critic_rng, sample_obs)

        actor_params = self.actor.init(actor_rng, sample_obs, )

        self.actor_optimizer = \
            optax.apply_if_finite(optax.adamw(learning_rate=lr_actor, weight_decay=weight_decay_actor),
                                  10000000
                                  )
        actor_opt_state = self.actor_optimizer.init(actor_params)
        self.critic_optimizer = optax.apply_if_finite(
            optax.adamw(learning_rate=lr_critic, weight_decay=weight_decay_critic),
            10000000
        )
        critic_opt_state = self.critic_optimizer.init(critic_params)
        target_critic_params = deepcopy(critic_params)
        state_normalizer_state = self.state_normalizer.initialize_normalizer_state()
        reward_normalizer_state = self.reward_normalizer.initialize_normalizer_state()

        init_state_rng, rng = jax.random.split(rng, 2)
        self.init_state = BPTTState(
            actor_opt_state=actor_opt_state,
            actor_params=actor_params,
            critic_opt_state=critic_opt_state,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            state_normalizer_state=state_normalizer_state,
            reward_normalizer_state=reward_normalizer_state,
            key=init_state_rng
        )

        self.horizon = horizon
        self.num_samples_per_gradient_update = num_samples_per_gradient_update
        self.sample_simulated_transitions = sample_simulated_transitions
        self.normalize = normalize
        self.action_normalize = action_normalize
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.train_steps = train_steps
        self.reset_optimizer = reset_optimizer
        self.evaluate_agent = evaluation_frequency > 0
        self.evaluation_samples = evaluation_samples
        self.evaluation_horizon = evaluation_horizon
        self.evaluation_frequency = evaluation_frequency
        self.discount = discount
        self.lambda_ = lambda_
        self.tau = target_soft_update_tau
        self.use_best_trained_policy = use_best_trained_policy
        self.loss_ent_coefficient = loss_ent_coefficient
        self.critic_updates_per_policy_updates = critic_updates_per_policy_update
        self.train_policy = lambda opt_state, obs: self.act(opt_state, obs, evaluate=False)
        dummy_transition = Transition(
            observation=jnp.zeros(self.obs_dim),
            action=jnp.zeros(self.action_dim),
            next_observation=jnp.zeros(self.obs_dim),
            reward=jnp.zeros(1),
            discount=jnp.zeros(1),
        )
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(
            dummy_transition
        )

        self.sampling_buffer = UniformSamplingQueue(max_replay_size=10000000,
                                                    dummy_data_sample=dummy_transition,
                                                    sample_batch_size=self.num_samples_per_gradient_update)
        self.init_buff_state = self.sampling_buffer.init(rng)

        self._unflatten_fn = jax.vmap(self._unflatten_fn)

    def init_optimizer_state(self):
        return self.init_state

    def update_normalizers(self, transition: Transition, bptt_state: BPTTState):
        state_normalizer_state = self.state_normalizer.update(transition.observation, bptt_state.state_normalizer_state)
        reward_normalizer_state = self.reward_normalizer.update(transition.reward,
                                                                bptt_state.reward_normalizer_state)
        new_bptt_state = bptt_state.replace(state_normalizer_state=state_normalizer_state,
                                            reward_normalizer_state=reward_normalizer_state)
        return new_bptt_state

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def act(self, opt_state: BPTTState, obs: chex.Array, evaluate: bool = True, *args, **kwargs) -> \
            Tuple[chex.Array, BPTTState]:

        normalized_obs = self.state_normalizer.normalize(obs, opt_state.state_normalizer_state)
        mu, sig = self.actor.apply(opt_state.actor_params, normalized_obs)

        def squash_action(x: chex.Array) -> chex.Array:
            squashed_action = nn.tanh(x)
            # sanity check to clip between -1 and 1
            squashed_action = jnp.clip(squashed_action, -0.999, 0.999)
            return squashed_action

        if evaluate:
            return squash_action(mu), opt_state
        else:
            key = opt_state.key
            sample_key, key = jax.random.split(key, 2)
            new_opt_state = opt_state.replace(key=key)
            action = mu + jax.random.normal(sample_key, mu.shape) * sig
            return squash_action(action), new_opt_state

    def actor_loss(self, init_state: chex.Array, bptt_state: BPTTState, system_params: SystemParams):
        chex.assert_shape(init_state, (self.obs_dim,))
        trajectory = rollout_policy(
            system=self.system,
            system_params=system_params,
            init_state=init_state,
            policy=self.train_policy,
            policy_state=bptt_state,
            horizon=self.horizon,
            stop_grads=True,
        )
        next_obs = trajectory.next_observation
        next_obs = self.state_normalizer.normalize(next_obs, bptt_state.state_normalizer_state)
        reward = trajectory.reward
        reward = self.reward_normalizer.normalize(reward, bptt_state.reward_normalizer_state)
        v_1, v_2 = self.critic.apply(bptt_state.target_critic_params, next_obs)
        bootstrap_values = jnp.minimum(v_1, v_2)
        lambda_values = lambda_return(reward, bootstrap_values, self.discount, self.lambda_)
        obs = self.state_normalizer.normalize(trajectory.observation, bptt_state.state_normalizer_state)
        pcont = jnp.ones(self.horizon)
        pcont = pcont.at[1:].set(self.discount)
        disc = jnp.cumprod(pcont)
        log_prob = self.actor.get_log_prob(squashed_action=trajectory.action, obs=obs,
                                           params=bptt_state.actor_params)
        entropy_loss = -log_prob.mean()
        actor_loss = -(lambda_values * disc).mean() + entropy_loss * self.loss_ent_coefficient
        return actor_loss, entropy_loss, lambda_values, trajectory

    def _train_step(self, initial_states: chex.Array, bptt_state: BPTTState, system_params: SystemParams):

        sampling_key, key = jax.random.split(bptt_state.key, 2)
        sys_sampling_key, sys_key = jax.random.split(system_params.key, 2)

        def actor_loss_fn(params):
            opt_state = bptt_state.replace(actor_params=params, key=sampling_key)
            sys_state = system_params.replace(key=sys_sampling_key)
            actor_loss, entropy_loss, lambda_values, trajectory = jax.vmap(self.actor_loss, in_axes=(0, None, None))(
                initial_states, opt_state, sys_state)

            def flatten_array(x):
                out = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x.reshape(-1)
                return out

            trajectory = jax.tree_util.tree_map(flatten_array, trajectory)
            lambda_values = lambda_values.reshape(-1)
            return actor_loss.mean(), (trajectory, lambda_values, entropy_loss.mean())

        rest, grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            bptt_state.actor_params
        )
        actor_loss, (trajectories, lambda_values, entropy_loss) = rest
        updates, new_actor_opt_state = self.actor_optimizer.update(grads, bptt_state.actor_opt_state,
                                                                   params=bptt_state.actor_params)
        new_actor_params = optax.apply_updates(bptt_state.actor_params, updates)
        actor_grad_norm = optax.global_norm(grads)

        critic_training_key, key = jax.random.split(key, 2)
        num_transitions = initial_states.shape[0] * self.horizon
        batch_size = math.ceil(num_transitions / self.critic_updates_per_policy_updates)
        transition_indices = jax.random.randint(critic_training_key, minval=0, maxval=num_transitions,
                                                shape=(self.critic_updates_per_policy_updates, batch_size))
        shuffled_transitions = jax.tree_util.tree_map(lambda x: x[transition_indices], trajectories)
        shuffled_lambda = lambda_values.reshape(-1)[transition_indices]

        def update_critic(carry, ins):
            critic_params, critic_opt_state, target_critic_params = carry[0], carry[1], carry[2]
            traj, lamb = ins[0], ins[1]

            def critic_loss_fn(params) -> jax.Array:
                obs = self.state_normalizer.normalize(traj.observation, bptt_state.state_normalizer_state)
                v_1, v_2 = self.critic.apply(params, obs)
                v_loss = 0.5 * (l2_loss(v_1, lamb).mean() + l2_loss(v_2, lamb).mean())
                return v_loss
                # return v_loss.mean()

            critic_loss, grads = jax.value_and_grad(critic_loss_fn, has_aux=False)(critic_params)
            updates, new_critic_opt_state = self.critic_optimizer.update(grads, critic_opt_state,
                                                                         params=critic_params)
            new_critic_params = optax.apply_updates(critic_params, updates)
            critic_grad_norm = optax.global_norm(grads)
            new_target_params = soft_update(target_critic_params, new_critic_params, tau=self.tau)
            outs = [critic_loss, critic_grad_norm]
            carry = [new_critic_params, new_critic_opt_state, new_target_params]
            return carry, outs

        carry = [bptt_state.critic_params, bptt_state.critic_opt_state, bptt_state.target_critic_params]
        carry, outs = jax.lax.scan(update_critic, carry, xs=[shuffled_transitions, shuffled_lambda],
                                   length=self.critic_updates_per_policy_updates)
        new_critic_params, new_critic_opt_state, new_target_critic_params = carry[0], carry[1], carry[2]
        critic_loss = outs[0][-1]
        critic_grad_norm = outs[1][-1]

        new_bptt_state = bptt_state.replace(
            actor_params=new_actor_params,
            actor_opt_state=new_actor_opt_state,
            key=key,
            critic_opt_state=new_critic_opt_state,
            critic_params=new_critic_params,
            target_critic_params=new_target_critic_params,
        )
        summary = BPTTAgentSummary(
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
        )
        system_params = system_params.replace(key=sys_key)
        return new_bptt_state, summary, trajectories, system_params

    @functools.partial(jax.jit, static_argnums=(0,))
    def train(self,
              buffer_state: ReplayBufferState,
              bptt_state: BPTTState,
              system_params: SystemParams,
              *args,
              **kwargs,
              ) -> BPTTTrainingOutput:
        train_key, key = jax.random.split(bptt_state.key, 2)
        eval_rng, train_key = jax.random.split(train_key, 2)
        eval_idx = jax.random.randint(eval_rng, (self.evaluation_samples,), minval=buffer_state.sample_position,
                                      maxval=buffer_state.insert_position)
        eval_data = jnp.take(buffer_state.data, eval_idx, axis=0, mode='wrap')
        eval_transitions = self._unflatten_fn(eval_data)
        eval_obs = eval_transitions.observation
        eval_sim_key, buffer_key, train_key = jax.random.split(train_key, 3)
        train_bptt_state = bptt_state.replace(key=train_key)

        transitions = self._unflatten_fn(buffer_state.data)
        train_buffer_state = self.sampling_buffer.insert(self.init_buff_state, transitions)

        def sample_obs(buff_state):
            new_buff_state, initial_transitions = self.sampling_buffer.sample(buff_state)
            initial_obs = initial_transitions.observation
            return initial_obs, new_buff_state

        def step(carry, ins):
            opt_state, best_opt_state, system_params, buffer_key, prev_summary = carry[0], carry[1], carry[2], \
                carry[3], carry[4]
            buff_state = carry[5]
            prev_reward = prev_summary.reward
            best_reward = prev_summary.best_reward

            initial_obs, new_buff_state = sample_obs(buff_state)
            new_opt_state, summary, transitions, new_system_params = self._train_step(
                initial_states=initial_obs,
                bptt_state=opt_state,
                system_params=system_params,
            )
            if self.normalize:
                new_opt_state = self.update_normalizers(transitions, bptt_state=new_opt_state)
            if self.sample_simulated_transitions:
                new_buff_state = self.sampling_buffer.insert(new_buff_state, transitions)
            if self.evaluate_agent:
                def evaluate_policy():
                    def rollout(obs):
                        trajectory = rollout_policy(
                            system=self.system,
                            system_params=system_params,
                            init_state=obs,
                            policy=self.act,
                            policy_state=new_opt_state,
                            horizon=self.evaluation_horizon,
                            stop_grads=True,
                        )
                        return trajectory

                    trajectory = jax.vmap(rollout)(eval_obs)
                    reward = trajectory.reward.sum(axis=-1).mean()

                    def get_new_reward():
                        return reward, new_opt_state,

                    def get_prev_best_reward():
                        return best_reward, best_opt_state

                    new_best_reward, new_best_opt_state = jax.lax.cond(
                        reward > best_reward,
                        get_new_reward,
                        get_prev_best_reward,
                    )
                    return reward, new_best_reward, new_best_opt_state

                def skip_evaluation():
                    return prev_reward, best_reward, best_opt_state

                reward, new_best_reward, new_best_opt_state = \
                    jax.lax.cond(jnp.logical_or(ins % self.evaluation_frequency == 0, ins == self.train_steps - 1),
                                 evaluate_policy,
                                 skip_evaluation
                                 )
            else:
                reward = prev_reward
                new_best_reward, new_best_opt_state = reward, new_opt_state
            summary = summary.replace(reward=reward, best_reward=new_best_reward)
            carry = [new_opt_state, new_best_opt_state, new_system_params, key, summary, new_buff_state]
            outs = summary
            return carry, outs

        carry = [train_bptt_state, train_bptt_state, system_params, buffer_key, BPTTAgentSummary(),
                 train_buffer_state]
        xs = jnp.arange(self.train_steps)
        carry, outs = jax.lax.scan(step, carry, xs=xs, length=self.train_steps)

        if self.use_best_trained_policy:
            trained_state = carry[1]
        else:
            trained_state = carry[0]
        output = BPTTTrainingOutput(
            bptt_state=trained_state,
            bptt_summary=outs,
        )
        return output
