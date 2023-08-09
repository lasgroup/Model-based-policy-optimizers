from typing import Callable, Optional

import chex
import jax
import optax
import jax.numpy as jnp


def loss_and_pgrad(loss_fn: Callable[..., chex.Array],
                   pmap_axis_name: Optional[str],
                   has_aux: bool = False):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grad = g(*args, **kwargs)
        return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def gradient_update_fn(loss_fn: Callable[..., chex.Array],
                       optimizer: optax.GradientTransformation,
                       pmap_axis_name: Optional[str],
                       has_aux: bool = False):
    """Wrapper of the loss function that apply gradient updates.

    Args:
      loss_fn: The loss function.
      optimizer: The optimizer to apply gradients.
      pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
      has_aux: Whether the loss_fn has auxiliary data.

    Returns:
      A function that takes the same argument as the loss function plus the
      optimizer state. The output of this function is the loss, the new parameter,
      and the new optimizer state.
    """
    loss_and_pgrad_fn = loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

    def f(*args, optimizer_state):
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state, params=args[0])
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state

    return f


def metrics_to_float(my_dict: dict) -> dict:
    for key, value in my_dict.items():
        my_dict[key] = float(value)
    return my_dict


def tree_stack(trees, axis=0):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l, axis=axis) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)
