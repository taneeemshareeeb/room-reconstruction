#!/usr/bin/env python3
"""Compare an experimental outlier-filtered cloud with its original."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from room_reconstruction.pointcloud import (
    load_archive_point_cloud,
    statistical_outlier_distances,
)


DEFAULT_CAPTURE = Path("data/raw/single_scan_with_ceiling.zip")
DEFAULT_OUTPUT = Path("reports/single_scan_with_ceiling_filter_strengths.png")
STD_RATIOS = (1.5, 2.5, 3.5)


def compare_clouds(
    capture_path: Path,
    output_path: Path,
    *,
    input_path: Path | None = None,
    frame_step: int = 10,
    voxel_size: float = 0.02,
    neighbors: int = 16,
) -> tuple[int, list[tuple[float, int]]]:
    if input_path is None:
        original = load_archive_point_cloud(
            capture_path, frame_step=frame_step, voxel_size=voxel_size
        )
    else:
        with np.load(input_path) as archive:
            if "points" not in archive:
                raise ValueError(f"{input_path} does not contain a 'points' array")
            original = np.asarray(archive["points"], dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or len(original) == 0:
        source = input_path or capture_path
        raise ValueError(f"{source} must produce a non-empty N x 3 point cloud")
    neighbor_distances = statistical_outlier_distances(
        original, neighbors=neighbors
    )
    center = float(neighbor_distances.mean())
    spread = float(neighbor_distances.std())
    variants = [
        (ratio, original[neighbor_distances <= center + ratio * spread])
        for ratio in STD_RATIOS
    ]

    x_limits = (float(original[:, 0].min()), float(original[:, 0].max()))
    z_limits = (float(original[:, 2].min()), float(original[:, 2].max()))
    y_limits = (float(original[:, 1].min()), float(original[:, 1].max()))
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True,
                             constrained_layout=True)
    clouds = [original, *(cloud for _, cloud in variants)]
    titles = [f"Original · {len(original):,} points"]
    titles.extend(
        f"{ratio:g}σ · {len(cloud):,} points · "
        f"removed {(len(original) - len(cloud)) / len(original) * 100:.2f}%"
        for ratio, cloud in variants
    )
    scatter = None
    for axis, points, title in zip(axes.ravel(), clouds, titles):
        stride = max(1, len(points) // 250_000)
        display = points[::stride]
        scatter = axis.scatter(
            display[:, 0], display[:, 2], c=display[:, 1], s=0.25,
            cmap="viridis", vmin=y_limits[0], vmax=y_limits[1],
            linewidths=0, rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlim(x_limits)
        axis.set_ylim(z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("World x (m)")
        axis.grid(alpha=0.15, linewidth=0.5)
    axes[0, 0].set_ylabel("World z (m)")
    axes[1, 0].set_ylabel("World z (m)")
    fig.suptitle(
        f"Statistical outlier filter · {neighbors} neighbors\n"
        "Higher σ keeps more points · experimental cleanup"
    )
    assert scatter is not None
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="World height y (m)", shrink=0.85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return len(original), [(ratio, len(cloud)) for ratio, cloud in variants]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument(
        "--input", type=Path,
        help="use a saved point-cloud .npz instead of loading the capture ZIP",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--neighbors", type=int, default=16)
    args = parser.parse_args()
    original_count, variants = compare_clouds(
        args.capture, args.output, input_path=args.input,
        frame_step=args.frame_step, voxel_size=args.voxel_size,
        neighbors=args.neighbors,
    )
    print(f"original points: {original_count:,}")
    for ratio, retained in variants:
        removed = original_count - retained
        percent = removed / original_count * 100 if original_count else 0.0
        print(f"{ratio:g}σ: {retained:,} retained; {removed:,} removed ({percent:.2f}%)")
    print(f"comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
