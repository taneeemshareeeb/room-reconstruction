#!/usr/bin/env python3
"""Render a top-down comparison of sampled point-cloud settings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from room_reconstruction.pointcloud import load_capture_point_cloud


STEPS = (10, 20, 40)
VOXELS = (0.02, 0.05)


def create_comparison(capture_root: Path, output_path: Path) -> None:
    clouds: dict[tuple[int, float], np.ndarray] = {}
    timings: dict[tuple[int, float], float] = {}
    for voxel in VOXELS:
        for step in STEPS:
            started = time.perf_counter()
            cloud = load_capture_point_cloud(
                capture_root,
                frame_step=step,
                voxel_size=voxel,
            )
            timings[(step, voxel)] = time.perf_counter() - started
            clouds[(step, voxel)] = cloud
            print(
                f"step={step:2d} voxel={voxel:.2f}m "
                f"points={len(cloud):,} seconds={timings[(step, voxel)]:.2f}"
            )

    all_points = np.concatenate(list(clouds.values()), axis=0)
    x_limits = (float(all_points[:, 0].min()), float(all_points[:, 0].max()))
    z_limits = (float(all_points[:, 2].min()), float(all_points[:, 2].max()))
    y_limits = (float(all_points[:, 1].min()), float(all_points[:, 1].max()))

    fig, axes = plt.subplots(
        len(VOXELS),
        len(STEPS),
        figsize=(15, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle("Single room: top-down point-cloud comparison", fontsize=17)
    scatter = None
    for row, voxel in enumerate(VOXELS):
        for col, step in enumerate(STEPS):
            axis = axes[row, col]
            points = clouds[(step, voxel)]
            scatter = axis.scatter(
                points[:, 0],
                points[:, 2],
                c=points[:, 1],
                s=0.25,
                cmap="viridis",
                vmin=y_limits[0],
                vmax=y_limits[1],
                linewidths=0,
                rasterized=True,
            )
            axis.set_title(
                f"Every {step}th frame · {voxel * 100:.0f} cm voxels\n"
                f"{len(points):,} points · {timings[(step, voxel)]:.2f} s"
            )
            axis.set_xlim(x_limits)
            axis.set_ylim(z_limits)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("World x (m)")
            if col == 0:
                axis.set_ylabel("World z (m)")
            axis.grid(alpha=0.15, linewidth=0.5)

    assert scatter is not None
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="World height y (m)", shrink=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture_root",
        nargs="?",
        type=Path,
        default=Path("data/processed/single_room/c00a170fe1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pointcloud_parameter_comparison.png"),
    )
    args = parser.parse_args()
    create_comparison(args.capture_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
