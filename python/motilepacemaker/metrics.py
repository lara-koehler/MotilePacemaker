"""Analysis metrics for source trajectories and field snapshots."""

import numpy as np


def unwrap_periodic_1d(x, L):
    """Unwrap a periodic 1D trajectory (shape (T,)), assuming steps smaller than L/2."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    dx[dx < -L / 2] += L
    dx[dx > L / 2] -= L
    return np.concatenate(([x[0]], x[0] + np.cumsum(dx)))


def unwrap_periodic_trajectories(x_source, L):
    """Unwrap an (N_sources, T) array of periodic positions along axis 1."""
    return np.stack([unwrap_periodic_1d(row, L) for row in x_source])


def mean_squared_displacement(x_unwrapped, y_unwrapped, max_lag=None):
    """MSD(lag), averaged over sources and start times, from unwrapped
    trajectories of shape (N_sources, T)."""
    n_steps = x_unwrapped.shape[1]
    if max_lag is None:
        max_lag = n_steps // 2
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        dx = x_unwrapped[:, lag:] - x_unwrapped[:, :-lag]
        dy = y_unwrapped[:, lag:] - y_unwrapped[:, :-lag]
        msd[lag - 1] = np.mean(dx**2 + dy**2)
    return msd


def field_phase(u, v, u0=0.0, v0=0.0):
    """Approximate oscillator phase as the polar angle of (u, v) around
    (u0, v0). A first-pass estimate (not a true asymptotic phase), valid away
    from the fixed point -- good enough for a Kuramoto-style order parameter."""
    return np.arctan2(v - v0, u - u0)


def kuramoto_order_parameter(phases):
    """`phases`: array of shape (N_oscillators, ...). Returns
    |<exp(i*phase)>| averaged over axis 0 (the oscillator axis)."""
    phases = np.asarray(phases)
    return np.abs(np.mean(np.exp(1j * phases), axis=0))


def kuramoto_order_parameter_over_time(u_source, v_source, u0=None, v0=None):
    """Kuramoto order parameter r(t) for a set of source (u, v) trajectories
    of shape (N_sources, T). If `u0`/`v0` aren't given, centers the phase on
    the mean of `u_source`/`v_source` over all sources and time -- a simple,
    robust proxy for the center of the oscillators' limit cycle (rather than
    an arbitrary fixed point like (0, 0), which may not sit inside the orbit).
    """
    u0 = np.mean(u_source) if u0 is None else u0
    v0 = np.mean(v_source) if v0 is None else v0
    phases = field_phase(u_source, v_source, u0=u0, v0=v0)
    return kuramoto_order_parameter(phases)


def density_fluctuations(positions, L, window_sizes, n_samples=200, rng=None):
    """Variance of particle counts in randomly placed square windows of each
    size in `window_sizes`, on a periodic domain of side `L`. A general,
    cheap proxy for spatial order: sub-linear growth of variance with window
    area signals suppressed density fluctuations (hyperuniform-like);
    variance growing linearly with area is Poissonian.

    `positions`: (N, 2) array.
    """
    rng = np.random.default_rng() if rng is None else rng
    positions = np.asarray(positions)
    variances = np.zeros(len(window_sizes))
    for k, w in enumerate(window_sizes):
        centers = rng.uniform(0, L, size=(n_samples, 2))
        counts = np.zeros(n_samples)
        for s, c in enumerate(centers):
            dx = np.abs(((positions[:, 0] - c[0] + L / 2) % L) - L / 2)
            dy = np.abs(((positions[:, 1] - c[1] + L / 2) % L) - L / 2)
            counts[s] = np.sum((dx < w / 2) & (dy < w / 2))
        variances[k] = np.var(counts)
    return variances
