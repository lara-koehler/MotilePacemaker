#!/usr/bin/env python3
"""
Usage: python scripts/analyze_regime_transitions.py <output.h5> [out_dir] [--stable-frac 0.15]

Diagnoses what's driving alternation between a spatially-uniform ("stable")
field and bursts/waves, and what's happening to the particle lattice while it
does, from a single raw run's .h5 -- designed to run *on the cluster* next to
the raw file (never needs the full u_field_stack in memory: field statistics
are streamed frame-by-frame via `io.field_stats_over_time`) and produce only
a small PNG + .npz to copy back, not the raw file itself.

Four time series, all at their own natural cadence:
  - mean/std of u(x,y) over the whole field, per saved field frame -- std
    near 0 means spatially uniform ("stable"); elevated std flags a
    wave/burst. `--stable-frac` (default 0.15) sets the fraction of the
    run's peak std below which a frame counts as "stable", used only to
    print interval-duration statistics.
  - psi4, psi6: n-fold bond-orientational order of the particle lattice
    (metrics.bond_orientational_order) at each saved source frame -- which
    packing symmetry (square vs. hexagonal) dominates, and whether that
    changes between stable and burst epochs.
  - nearest-neighbor distances (metrics.nearest_neighbor_distances) at each
    saved source frame: the minimum over all particles (worst-case defect
    depth) and a count of pairs closer than the mechanical model's own
    `rcut_near_factor * interaction_range` -- inside that radius, the
    long-range field-gated force is switched off entirely (see
    sources.jl's `gated_force_components`), so a pair that ends up this
    close is only held apart by short-range WCA repulsion, "orphaned" from
    whatever is organizing the rest of the lattice.
  - Kuramoto order parameter over time (metrics.kuramoto_order_parameter_over_time),
    for cross-reference against the other three.
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from motilepacemaker import io, metrics


def parse_args(argv):
    positional = []
    stable_frac = 0.15
    i = 0
    while i < len(argv):
        if argv[i] == "--stable-frac":
            stable_frac = float(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return positional, stable_frac


def stable_interval_stats(field_times, std_u, stable_frac):
    """Segment `field_times` into "stable" (std_u below `stable_frac` of its
    own peak) and "burst" runs; return (stable_durations, burst_durations),
    both in the run's own time units."""
    threshold = stable_frac * np.max(std_u)
    is_stable = std_u < threshold
    durations = {True: [], False: []}
    dt_field = field_times[1] - field_times[0] if len(field_times) > 1 else 0.0
    run_start = 0
    for i in range(1, len(is_stable) + 1):
        if i == len(is_stable) or is_stable[i] != is_stable[run_start]:
            durations[is_stable[run_start]].append((i - run_start) * dt_field)
            run_start = i
    return durations[True], durations[False], threshold


def main():
    positional, stable_frac = parse_args(sys.argv[1:])
    if len(positional) < 1:
        print("Usage: analyze_regime_transitions.py <output.h5> [out_dir] [--stable-frac 0.15]")
        sys.exit(1)

    h5path = positional[0]
    run_name = io.run_name(h5path)
    out_dir = Path(positional[1]) if len(positional) > 1 else Path("../data/processed") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"analyze_regime_transitions: {h5path}")
    field_times, mean_u, std_u = io.field_stats_over_time(h5path, verbose=True)

    data = io.load_run(h5path, keys=["x_source", "y_source", "u_source", "v_source"], verbose=True)
    params = data["params"]
    L = params["grid"]["L"]
    interaction_range = params["mechanics"]["interaction_range"]
    rcut_near_factor = params["mechanics"].get("rcut_near_factor", 0.05)
    defect_radius = rcut_near_factor * interaction_range

    x_source, y_source = data["x_source"], data["y_source"]
    n_sources, n_source_saved = x_source.shape

    print(f"computing lattice order + neighbor stats over {n_source_saved} source frames...")
    psi4 = np.empty(n_source_saved)
    psi6 = np.empty(n_source_saved)
    min_nn_dist = np.empty(n_source_saved)
    defect_count = np.empty(n_source_saved)
    report_every = max(1, n_source_saved // 20)
    for t in range(n_source_saved):
        if t % report_every == 0:
            print(f"\r  frame {t}/{n_source_saved}", end="", flush=True)
        positions = np.stack([x_source[:, t], y_source[:, t]], axis=1)
        psi4[t] = metrics.bond_orientational_order(positions, L, n=4, k=4)
        psi6[t] = metrics.bond_orientational_order(positions, L, n=6, k=6)
        nn = metrics.nearest_neighbor_distances(positions, L)
        min_nn_dist[t] = nn.min()
        defect_count[t] = np.sum(nn < defect_radius)
    print(f"\r  frame {n_source_saved}/{n_source_saved}")

    kuramoto_r = metrics.kuramoto_order_parameter_over_time(data["u_source"], data["v_source"])
    source_times = data["times"]

    stable_durations, burst_durations, std_threshold = stable_interval_stats(
        field_times, std_u, stable_frac
    )
    print(f"\nstable-phase segmentation (std_u < {stable_frac} * peak = {std_threshold:.4g}):")
    if stable_durations:
        print(f"  {len(stable_durations)} stable interval(s), "
              f"duration mean={np.mean(stable_durations):.3g}, "
              f"median={np.median(stable_durations):.3g}, "
              f"min={np.min(stable_durations):.3g}, max={np.max(stable_durations):.3g}")
    else:
        print("  no stable intervals found")
    if burst_durations:
        print(f"  {len(burst_durations)} burst interval(s), "
              f"duration mean={np.mean(burst_durations):.3g}, "
              f"median={np.median(burst_durations):.3g}")

    n_defect_frames = int(np.sum(defect_count > 0))
    print(f"\ndefect pairs (nearest-neighbor dist < {defect_radius:.4g}, "
          f"= rcut_near_factor * interaction_range): present in "
          f"{n_defect_frames}/{n_source_saved} saved frames "
          f"({100 * n_defect_frames / n_source_saved:.1f}%)")

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    axes[0].plot(field_times, mean_u, label="mean u")
    axes[0].fill_between(field_times, mean_u - std_u, mean_u + std_u, alpha=0.3, label="+/- std u")
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("field u")
    axes[0].set_title(f"{run_name}: spatial mean/std of u(x,y,t)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(field_times, std_u, color="C3")
    axes[1].axhline(std_threshold, color="grey", ls="--", lw=1,
                     label=f"stable threshold ({stable_frac}x peak)")
    axes[1].set_ylabel("std(u)")
    axes[1].set_title("spatial std of u -- low = spatially uniform (\"stable\"), high = wave/burst")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(source_times, psi4, label="psi4 (square)")
    axes[2].plot(source_times, psi6, label="psi6 (hexagonal)")
    axes[2].set_ylabel("bond-orientational order (psi4, psi6)")
    #axes[2].set_ylim(0, 1.05)
    ax2b = axes[2].twinx()
    ax2b.plot(source_times, kuramoto_r, color="grey", alpha=0.6, label="Kuramoto r")
    ax2b.set_ylabel("Kuramoto r", color="grey")
    ax2b.set_ylim(0, 1.05)
    axes[2].set_title("lattice bond-orientational order (left) vs. Kuramoto order (right)")
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].plot(source_times, min_nn_dist, label="min NN distance")
    axes[3].axhline(defect_radius, color="grey", ls="--", lw=1, label="defect radius (rcut_near)")
    axes[3].set_yscale("log")
    ax3b = axes[3].twinx()
    ax3b.plot(source_times, defect_count, color="C1", alpha=0.6, label="# defect pairs")
    ax3b.set_ylabel("# defect pairs", color="C1")
    axes[3].set_ylabel("min NN distance")
    axes[3].set_xlabel("time")
    axes[3].set_title("nearest-neighbor distance / defect pairs over time")
    axes[3].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out_png = out_dir / f"{run_name}_regime_diagnostics.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")

    out_npz = out_dir / f"{run_name}_regime_diagnostics.npz"
    np.savez(
        out_npz,
        field_times=field_times, mean_u=mean_u, std_u=std_u,
        source_times=source_times, psi4=psi4, psi6=psi6, kuramoto_r=kuramoto_r,
        min_nn_dist=min_nn_dist, defect_count=defect_count, defect_radius=defect_radius,
        stable_durations=np.array(stable_durations), burst_durations=np.array(burst_durations),
    )
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
