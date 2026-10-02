#!/usr/bin/env python3
"""Compare an experimental outlier-filtered cloud with its original."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from room_reconstruction.pointcloud import statistical_outlier_filter
from room_reconstruction.pointcloud import load_archive_point_cloud


DEFAULT_CAPTURE = Path("data/raw/single_scan_with_ceiling.zip")
DEFAULT_OUTPUT = Path("reports/single_scan_with_ceiling_filter_comparison.png")


def compare_clouds(
    capture_path: Path,
    output_path: Path,
    *,
    input_path: Path | None = None,
    frame_step: int = 10,
    voxel_size: float = 0.02,
    neighbors: int = 16,
    std_ratio: float = 2.5,
) -> tuple[int, int]:
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
        raise ValueError(f"{input_path} must contain a non-empty N x 3 point cloud")
    filtered = statistical_outlier_filter(
        original, neighbors=neighbors, std_ratio=std_ratio
    )

    all_points = original
    x_limits = (float(all_points[:, 0].min()), float(all_points[:, 0].max()))
    z_limits = (float(all_points[:, 2].min()), float(all_points[:, 2].max()))
    y_limits = (float(all_points[:, 1].min()), float(all_points[:, 1].max()))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True,
                             constrained_layout=True)
    clouds = (original, filtered)
    titles = (
        f"Original · {len(original):,} points",
        f"Filtered · {len(filtered):,} points",
    )
    scatter = None
    for axis, points, title in zip(axes, clouds, titles):
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
    axes[0].set_ylabel("World z (m)")
    fig.suptitle(
        f"Statistical outlier filter · {neighbors} neighbors · "
        f"{std_ratio:g} standard deviations\n"
        "Experimental cleanup — compare with the original before using"
    )
    assert scatter is not None
    fig.colorbar(scatter, ax=axes.tolist(), label="World height y (m)", shrink=0.85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return len(original), len(filtered)


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
    parser.add_argument("--std-ratio", type=float, default=2.5)
    args = parser.parse_args()
    original_count, filtered_count = compare_clouds(
        args.capture, args.output, input_path=args.input,
        frame_step=args.frame_step, voxel_size=args.voxel_size,
        neighbors=args.neighbors, std_ratio=args.std_ratio,
    )
    removed = original_count - filtered_count
    percent = removed / original_count * 100 if original_count else 0.0
    print(f"original points: {original_count:,}")
    print(f"filtered points: {filtered_count:,}")
    print(f"removed: {removed:,} ({percent:.2f}%)")
    print(f"comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
