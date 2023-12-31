from mbpo.optimizers import iCemTO, iCemParams
from mbpo.systems import PendulumSystem
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
optimizer_key, init_key, key = jax.random.split(key, 3)
system = PendulumSystem()
system_state = system.reset(key)
system_params = system_state.system_params
opt_params = iCemParams()
cem_optimizer = iCemTO(horizon=20, action_dim=1, system=None, opt_params=opt_params, key=optimizer_key)
cem_optimizer.set_system(system)
cem_optimizer_state = cem_optimizer.init(init_key)

horizon = 200


def rollout_cem(carry, ins):
    system_state, cem_optimizer_state = carry[0], carry[1]
    action, new_cem_optimizer_state = cem_optimizer.act(obs=system_state.x_next, opt_state=cem_optimizer_state)
    new_system_state = system.step(x=system_state.x_next, u=action,
                                   system_params=system_state.system_params)
    new_cem_optimizer_state = new_cem_optimizer_state.replace(system_params=new_system_state.system_params)

    carry = [new_system_state, new_cem_optimizer_state]
    outs = [new_system_state.x_next, new_system_state.reward]
    return carry, outs


carry = [system_state, cem_optimizer_state]
carry, outs = jax.lax.scan(rollout_cem, carry, xs=None, length=200)

rewards = outs[-1]


def test_optimizer_performance():
    assert rewards.sum() >= -400
