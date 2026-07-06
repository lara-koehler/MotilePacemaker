#!/usr/bin/env python3
"""
Usage: python scripts/plot_kuramoto_comparison.py [out.png]

Plots the Kuramoto order parameter r(t) (computed from each source's
chemical (u, v) phase) over time for the wave, wave_2d, and chaotic runs, to
compare how synchronized/coherent their dynamics are. The legend shows only
the *real* mechanical/coupling/arrangement parameters that actually differ
across the three runs' saved configs (not the dimensionless chi/Gamma).

Defaults to reading data/raw/{wave,wave_2d,chaotic}/output.h5 and saving to
data/processed/kuramoto_comparison.png.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from motilepacemaker import io, metrics

RUN_NAMES = ["wave", "wave_2d", "chaotic"]
# sections/keys worth checking for differences across runs, in display order
CANDIDATE_PARAMS = [
    ("mechanics", "mobility"),
    ("mechanics", "interaction_strength"),
    ("mechanics", "sigma"),
    ("mechanics", "reciprocity"),
    ("coupling", "pacemaker_width"),
    ("sources", "arrangement"),
]


def differing_params(all_params):
    """Return the subset of CANDIDATE_PARAMS whose value isn't identical
    across all runs in `all_params` (a dict of name -> parsed TOML params)."""
    differing = []
    for section, key in CANDIDATE_PARAMS:
        values = [p.get(section, {}).get(key) for p in all_params.values()]
        if len(set(values)) > 1:
            differing.append((section, key))
    return differing


def legend_label(name, params, differing):
    parts = [name]
    for section, key in differing:
        value = params.get(section, {}).get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if "sources" in params and params["sources"].get("arrangement") == "grid_fill":
        parts.append(f"{params['sources']['grid_rows']}x{params['sources']['grid_cols']}")
    return ", ".join(parts)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "../data/processed/kuramoto_comparison.png"

    data = {}
    for name in RUN_NAMES:
        h5path = f"../data/raw/{name}/output.h5"
        data[name] = io.load_run(h5path)

    all_params = {name: d["params"] for name, d in data.items()}
    differing = differing_params(all_params)
    print("Real parameters that differ across runs:", differing)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in RUN_NAMES:
        d = data[name]
        r_t = metrics.kuramoto_order_parameter_over_time(d["u_source"], d["v_source"])
        label = legend_label(name, all_params[name], differing)
        ax.plot(d["times"], r_t, label=label, linewidth=1.2)

    ax.set_xlabel("time")
    ax.set_ylabel("Kuramoto order parameter r(t)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Source phase coherence across runs")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
