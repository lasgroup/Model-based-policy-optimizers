from typing import Callable, Optional
from functools import partial


import chex
import jax
import optax
import jax.numpy as jnp

from jax.numpy import sqrt, newaxis
from jax.numpy.fft import irfft, rfftfreq


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


@partial(jax.jit, static_argnums=(0, 1, 3))
def powerlaw_psd_gaussian(exponent: float, size: int, rng: jax.random.PRNGKey, fmin: float = 0) -> jax.Array:
    """
    # powerlaw_psd_gaussian: Generate colored noise. 
    Taken from: https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py

    Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.
    random_state :  int, numpy.integer, numpy.random.Generator, numpy.random.RandomState,
                    optional
        Optionally sets the state of NumPy's underlying random number generator.
        Integer-compatible values or None are passed to np.random.default_rng.
        np.random.RandomState or np.random.Generator are used directly.
        Default: None.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1. / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = jnp.sum(s_scale < fmin)  # Index of the cutoff

    def cutoff(x, idx):
        x_idx = jax.lax.dynamic_slice(x, start_indices=(idx,), slice_sizes=(1,))
        y = jnp.ones_like(x) * x_idx
        indexes = jnp.arange(0, x.shape[0], step=1)
        first_idx = indexes < idx
        z = (1 - first_idx) * x + first_idx * y
        return z

    def no_cutoff(x, idx):
        return x

    s_scale = jax.lax.cond(
        jnp.logical_and(ix < len(s_scale), ix),
        cutoff,
        no_cutoff,
        s_scale,
        ix
    )
    s_scale = s_scale ** (-exponent / 2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w = w.at[-1].set(w[-1] * (1 + (samples % 2)) / 2.)  # correct f = +-0.5
    sigma = 2 * sqrt(jnp.sum(w ** 2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    key_sr, key_si, rng = jax.random.split(rng, 3)
    sr = jax.random.normal(key=key_sr, shape=s_scale.shape) * s_scale
    si = jax.random.normal(key=key_si, shape=s_scale.shape) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * sqrt(2))  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * sqrt(2))  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    return y
