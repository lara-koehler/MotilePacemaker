#!/usr/bin/env python3
"""
Usage: python scripts/make_movie.py <output.h5> [out.mp4]

Requires ffmpeg to be installed and on PATH.
"""
import sys
from pathlib import Path

from motilepacemaker import io, viz


def main():
    if len(sys.argv) < 2:
        print("Usage: make_movie.py <output.h5> [out.mp4]")
        sys.exit(1)

    h5path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(h5path).with_name("movie.mp4"))

    data = io.load_run(h5path)
    params = data["params"]
    L = params["grid"]["L"]
    save_every = params["time"]["save_every"]
    source_save_every = params["time"].get("source_save_every", 1)

    vmax = float(abs(data["u_field_stack"]).max())
    viz.make_movie(data["u_field_stack"], data["x_source"], data["y_source"], L,
                    save_every, source_save_every, out_path, vmin=-vmax, vmax=vmax)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
