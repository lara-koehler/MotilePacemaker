"""Analysis metrics for source trajectories and field snapshots."""

import numpy as np
from scipy.spatial import cKDTree


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


def bond_orientational_order(positions, L, n=6, k=6):
    """n-fold bond-orientational order parameter psi_n for a periodic 2D
    point set (e.g. n=4 for square, n=6 for hexagonal packing): for each
    particle, average exp(i*n*theta) over the bond angles theta to its `k`
    nearest neighbors, then average that complex value over all particles
    and take the magnitude. Near 1 means the point set is locally n-fold
    coordinated everywhere and in phase (a near-perfect lattice of that
    symmetry); near 0 means disordered w.r.t. that symmetry (either a
    different symmetry, or no symmetry at all).

    `positions`: (N, 2). Periodic (minimum-image) neighbor search via
    `scipy.spatial.cKDTree`'s `boxsize`.
    """
    positions = np.asarray(positions) % L
    n_points = len(positions)
    tree = cKDTree(positions, boxsize=L)
    _, idx = tree.query(positions, k=k + 1)  # idx[:, 0] is each point itself

    psi = np.empty(n_points, dtype=complex)
    for i in range(n_points):
        dx = positions[idx[i, 1:]] - positions[i]
        dx -= L * np.round(dx / L)
        theta = np.arctan2(dx[:, 1], dx[:, 0])
        psi[i] = np.mean(np.exp(1j * n * theta))
    return np.abs(np.mean(psi))


def nearest_neighbor_distances(positions, L):
    """Minimum-image distance from each particle to its single nearest
    neighbor, on a periodic domain of side `L`. `positions`: (N, 2). Returns
    an (N,) array -- small values flag near-overlapping pairs (e.g. below a
    mechanical force's short-range cutoff, where two sources have fallen out
    of range of any longer-range organizing force and are held apart by
    short-range repulsion alone)."""
    positions = np.asarray(positions) % L
    tree = cKDTree(positions, boxsize=L)
    dist, _ = tree.query(positions, k=2)  # column 0 is self (distance 0)
    return dist[:, 1]


def local_field_values(positions, u_field, L):
    """Value of `u_field` at the grid cell nearest to each position, matching
    the Julia simulation's own sampling convention (`local_field_index` in
    sources.jl -- nearest-cell, not interpolated). `positions`: (N, 2).
    `u_field`: (nx, ny)."""
    positions = np.asarray(positions)
    nx, ny = u_field.shape
    ix = np.mod(np.round(positions[:, 0] / L * nx).astype(int), nx)
    iy = np.mod(np.round(positions[:, 1] / L * ny).astype(int), ny)
    return u_field[ix, iy]


def compute_mechanical_forces(positions, u_field, L, mech, dt=None):
    """Net mechanical force (WCA + field-gated) on each source, at one
    instant -- a direct, fully-vectorized (all N^2 pairs at once, no
    per-pair Python loop) port of julia/src/sources.jl's per-step force law
    (`compute_forces`/`accumulate_pair_force!`/`pair_raw_force`), for
    diagnostic plotting only. The actual simulation always runs in Julia;
    this never needs to be fast enough for a hot per-step loop, just correct.

    `positions`: (N, 2). `u_field`: (nx, ny), the field snapshot to sample
    (via `local_field_values`) and gate the attraction/repulsion sign on.
    `mech`: a run's resolved `[mechanics]` config dict (`params["mechanics"]`
    from `io.load_run`/`load_run_lite`) -- `sigma`, `epsilon_LJ`,
    `interaction_strength`, `interaction_range`, `u_threshold`,
    `reciprocity`, and optionally `rcut_far_factor` (default 5.0),
    `rcut_near_factor` (default 0.05), `max_force`.

    `dt`, if given and `mech` has no explicit `max_force`, reproduces the
    Julia side's own automatic default (`io.jl`: `0.5 * sigma / (dt *
    mobility)`) instead of leaving the force uncapped -- pass the run's
    `params["time"]["dt"]` for results that match the actual simulation.
    """
    positions = np.asarray(positions)
    N = positions.shape[0]
    local_u = local_field_values(positions, u_field, L)

    # dx[i, j] = xj - xi (periodic minimum image), matching sources.jl's own
    # `dx = positions[j] - positions[i]` convention
    dx = positions[None, :, 0] - positions[:, None, 0]
    dy = positions[None, :, 1] - positions[:, None, 1]
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    dist2 = dx**2 + dy**2
    np.fill_diagonal(dist2, np.inf)  # exclude self-interaction
    dist = np.sqrt(dist2)

    sigma, epsilon = mech["sigma"], mech["epsilon_LJ"]
    rc2_wca = (2 ** (1 / 6) * sigma) ** 2
    inv_r2 = 1.0 / dist2
    inv_r6 = sigma**6 * inv_r2**3
    inv_r12 = inv_r6**2
    f_wca = np.where(dist2 < rc2_wca, 24 * epsilon * inv_r2 * (2 * inv_r12 - inv_r6), 0.0)

    r0, strength = mech["interaction_range"], mech["interaction_strength"]
    rcut_far2 = (mech.get("rcut_far_factor", 5.0) * r0) ** 2
    rcut_near2 = (mech.get("rcut_near_factor", 0.05) * r0) ** 2
    gate_active = (dist2 < rcut_far2) & (dist2 > rcut_near2)

    if mech.get("reciprocity", "nonreciprocal") == "reciprocal":
        gate_u = 0.5 * (local_u[:, None] + local_u[None, :])
        sign = np.where(gate_u > mech["u_threshold"], 1.0, -1.0)
    else:
        # nonreciprocal: force on i (row i) is gated by u at i alone, so this
        # broadcasts row-wise rather than symmetrizing across the pair
        sign = np.where(local_u[:, None] > mech["u_threshold"], 1.0, -1.0) * np.ones_like(dist2)

    mag_gated = np.where(gate_active, sign * strength * np.exp(-dist / r0) / dist, 0.0)

    fx_raw = (f_wca + mag_gated) * dx
    fy_raw = (f_wca + mag_gated) * dy

    if "max_force" in mech:
        max_force = mech["max_force"]
    elif dt is not None:
        max_force = 0.5 * sigma / (dt * mech["mobility"])
    else:
        max_force = np.inf
    mag = np.sqrt(fx_raw**2 + fy_raw**2)
    scale = np.where(mag > max_force, max_force / np.where(mag > 0, mag, 1.0), 1.0)

    # sum over j of -(capped pair force), matching accumulate_pair_force!'s
    # `forces[i] -= fx` -- see this function's docstring for why this single
    # row-sum, with no separate i<j/i>j bookkeeping, reproduces both the
    # reciprocal and nonreciprocal cases exactly
    return np.stack([-(fx_raw * scale).sum(axis=1), -(fy_raw * scale).sum(axis=1)], axis=1)


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
