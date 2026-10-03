#!/usr/bin/env python3
"""Compare wall and room candidates across vertical point-cloud slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from room_reconstruction.geometry import detect_wall_candidates, estimate_floor_height
from room_reconstruction.pointcloud import load_archive_point_cloud
from room_reconstruction.rooms import detect_room_candidates


DEFAULT_CAPTURE = Path("data/raw/single_scan_with_ceiling.zip")
DEFAULT_OUTPUT = Path("reports/single_scan_with_ceiling_vertical_slices.png")
HEIGHT_BANDS = (
    ("Lower", 0.40, 0.80),
    ("Current", 0.85, 1.55),
    ("Upper", 1.50, 2.20),
    ("Combined", 0.40, 2.20),
)


def compare_slices(
    points: np.ndarray,
    output_path: Path,
    *,
    grid_size_m: float = 0.08,
    gap_closing_cells: int = 5,
) -> list[tuple[str, int, int, int, int]]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must be a non-empty N x 3 array")
    floor_y = estimate_floor_height(points)
    x_limits = (float(points[:, 0].min()), float(points[:, 0].max()))
    z_limits = (float(points[:, 2].min()), float(points[:, 2].max()))
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True,
                                constrained_layout=True)
    summary = []

    for axis, (label, lower, upper) in zip(axes.ravel(), HEIGHT_BANDS):
        in_slice = points[
            (points[:, 1] >= floor_y + lower)
            & (points[:, 1] <= floor_y + upper)
        ]
        walls = detect_wall_candidates(
            points,
            slice_min_height_m=lower,
            slice_max_height_m=upper,
            grid_size_m=grid_size_m,
        )
        rooms = detect_room_candidates(
            points,
            grid_size_m=grid_size_m,
            slice_min_height_m=lower,
            slice_max_height_m=upper,
            gap_closing_cells=gap_closing_cells,
        )
        occupied_cells = len(
            np.unique(np.floor(in_slice[:, (0, 2)] / grid_size_m).astype(np.int64), axis=0)
        ) if len(in_slice) else 0
        summary.append((label, len(in_slice), occupied_cells, len(walls), len(rooms)))

        display = in_slice[::max(1, len(in_slice) // 100_000)]
        axis.scatter(display[:, 0], display[:, 2], s=0.25, color="#87909a",
                     alpha=0.45, linewidths=0, rasterized=True)
        for wall in walls:
            start = wall["start_xz_m"]
            end = wall["end_xz_m"]
            axis.plot([start[0], end[0]], [start[1], end[1]], color="#d62728",
                      linewidth=1.6, alpha=0.85)
        for room in rooms:
            boundary = room["boundary_xz_m"]
            if len(boundary) >= 4:
                axis.plot([point[0] for point in boundary],
                          [point[1] for point in boundary],
                          color="#e67e22", linewidth=1.6, linestyle="--")
        axis.set_title(
            f"{label} · {floor_y + lower:.2f}–{floor_y + upper:.2f} m\n"
            f"{len(in_slice):,} pts · {occupied_cells:,} cells · "
            f"{len(walls)} wall / {len(rooms)} room candidates"
        )
        axis.set_xlim(x_limits)
        axis.set_ylim(z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("World x (m)")
        axis.grid(alpha=0.15, linewidth=0.5)

    axes[0, 0].set_ylabel("World z (m)")
    axes[1, 0].set_ylabel("World z (m)")
    figure.suptitle(
        f"Candidate stability across height slices · floor y={floor_y:.2f} m\n"
        "Red: wall traces · orange dashed: enclosed-room outlines · all unverified"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--input", type=Path, help="use a saved point-cloud .npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.02)
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

    summary = compare_slices(points, args.output)
    print(f"points: {len(points):,}")
    print("slice: points / occupied x-z cells / wall candidates / room candidates")
    for label, points_in_slice, cells, walls, rooms in summary:
        print(f"{label}: {points_in_slice:,} / {cells:,} / {walls} / {rooms}")
    print(f"comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
