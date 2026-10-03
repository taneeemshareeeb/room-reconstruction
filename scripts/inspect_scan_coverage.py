#!/usr/bin/env python3
"""Visualize where a capture contains points at several room heights."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from room_reconstruction.geometry import estimate_floor_height
from room_reconstruction.pointcloud import load_archive_point_cloud


DEFAULT_CAPTURE = Path("data/raw/single_scan_with_ceiling.zip")
DEFAULT_OUTPUT = Path("reports/single_scan_with_ceiling_coverage.png")
HEIGHT_BANDS = (
    ("Floor band", 0.00, 0.20),
    ("Lower wall band", 0.40, 0.80),
    ("Room detection band", 0.85, 1.55),
    ("Upper wall / ceiling band", 2.00, 2.40),
)


def coverage_cells(
    points: np.ndarray,
    *,
    floor_y: float,
    lower_height_m: float,
    upper_height_m: float,
    grid_size_m: float,
    min_points_per_cell: int,
) -> tuple[np.ndarray, int, int]:
    """Rasterize supported x-z cells inside one floor-relative height band."""
    selected = points[
        (points[:, 1] >= floor_y + lower_height_m)
        & (points[:, 1] < floor_y + upper_height_m)
    ]
    x_origin = float(np.floor(points[:, 0].min() / grid_size_m) * grid_size_m)
    z_origin = float(np.floor(points[:, 2].min() / grid_size_m) * grid_size_m)
    x_count = int(np.ceil((points[:, 0].max() - x_origin) / grid_size_m)) + 1
    z_count = int(np.ceil((points[:, 2].max() - z_origin) / grid_size_m)) + 1
    grid = np.zeros((x_count, z_count), dtype=np.uint32)
    if len(selected):
        x_indices = np.floor((selected[:, 0] - x_origin) / grid_size_m).astype(int)
        z_indices = np.floor((selected[:, 2] - z_origin) / grid_size_m).astype(int)
        np.add.at(grid, (x_indices, z_indices), 1)
    occupied = grid >= min_points_per_cell
    return occupied, int(len(selected)), int(occupied.sum())


def render_coverage(
    points: np.ndarray,
    output_path: Path,
    *,
    grid_size_m: float = 0.05,
    min_points_per_cell: int = 2,
) -> list[tuple[str, int, int, float]]:
    """Save one occupancy panel per height band and return coverage stats."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must be a non-empty N x 3 array")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates")
    if not np.isfinite(grid_size_m) or grid_size_m <= 0:
        raise ValueError("grid_size_m must be finite and greater than zero")
    if min_points_per_cell < 1:
        raise ValueError("min_points_per_cell must be at least 1")

    floor_y = estimate_floor_height(points)
    x_origin = float(np.floor(points[:, 0].min() / grid_size_m) * grid_size_m)
    z_origin = float(np.floor(points[:, 2].min() / grid_size_m) * grid_size_m)
    x_count = int(np.ceil((points[:, 0].max() - x_origin) / grid_size_m)) + 1
    z_count = int(np.ceil((points[:, 2].max() - z_origin) / grid_size_m)) + 1
    x_max = x_origin + x_count * grid_size_m
    z_max = z_origin + z_count * grid_size_m
    grid_area_cells = x_count * z_count

    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    stats = []
    for axis, (label, lower, upper) in zip(axes.ravel(), HEIGHT_BANDS):
        occupied, point_count, occupied_count = coverage_cells(
            points,
            floor_y=floor_y,
            lower_height_m=lower,
            upper_height_m=upper,
            grid_size_m=grid_size_m,
            min_points_per_cell=min_points_per_cell,
        )
        occupancy_percent = occupied_count / grid_area_cells * 100
        stats.append((label, point_count, occupied_count, occupancy_percent))
        axis.imshow(
            occupied.T,
            origin="lower",
            extent=(x_origin, x_max, z_origin, z_max),
            interpolation="nearest",
            cmap=ListedColormap(("#f5f5f5", "#176b87")),
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        axis.set_title(
            f"{label} · {floor_y + lower:.2f}–{floor_y + upper:.2f} m\n"
            f"{occupied_count:,} occupied cells ({occupancy_percent:.1f}%)"
        )
        axis.set_xlabel("World x (m)")
        axis.grid(alpha=0.12, linewidth=0.4)
    axes[0, 0].set_ylabel("World z (m)")
    axes[1, 0].set_ylabel("World z (m)")
    figure.suptitle(
        f"Point coverage by height · estimated floor y={floor_y:.2f} m\n"
        f"{grid_size_m * 100:.0f} cm cells · require {min_points_per_cell}+ points per cell"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--input", type=Path, help="use a saved point-cloud .npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument("--min-points-per-cell", type=int, default=2)
    args = parser.parse_args()

    if args.input is None:
        points = load_archive_point_cloud(
            args.capture, frame_step=args.frame_step, voxel_size=args.voxel_size
        )
    else:
        with np.load(args.input) as archive:
            if "points" not in archive:
                parser.error(f"{args.input} does not contain a 'points' array")
            points = np.asarray(archive["points"], dtype=np.float64)

    stats = render_coverage(
        points, args.output, grid_size_m=args.grid_size,
        min_points_per_cell=args.min_points_per_cell,
    )
    print(f"points: {len(points):,}")
    for label, point_count, cells, percent in stats:
        print(f"{label}: {point_count:,} points; {cells:,} occupied cells ({percent:.1f}%)")
    print(f"coverage image: {args.output}")
    print("Empty cells mean this height band has little or no recorded point support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
