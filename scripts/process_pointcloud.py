#!/usr/bin/env python3
"""Build and save a downsampled point cloud from one capture."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from room_reconstruction.pointcloud import (
    load_archive_point_cloud,
    load_capture_point_cloud,
)


def process_capture(
    capture_root: Path,
    output_path: Path,
    plot_path: Path,
    frame_step: int,
    voxel_size: float,
) -> None:
    capture_name = (
        capture_root.stem
        if capture_root.suffix.lower() == ".zip"
        else capture_root.parent.name
    )
    loader = (
        load_archive_point_cloud
        if capture_root.suffix.lower() == ".zip"
        else load_capture_point_cloud
    )
    points = loader(capture_root, frame_step=frame_step, voxel_size=voxel_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points,
        frame_step=np.int64(frame_step),
        voxel_size_m=np.float64(voxel_size),
    )

    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = axis.scatter(
        points[:, 0],
        points[:, 2],
        c=points[:, 1],
        s=0.35,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    axis.set_title(
        f"{capture_name.replace('_', ' ')} point cloud · every {frame_step}th frame · "
        f"{voxel_size * 100:g} cm voxels\n{len(points):,} points"
    )
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World z (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.18)
    figure.colorbar(image, ax=axis, label="World height y (m)")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"points: {len(points):,}")
    print(f"finite: {bool(np.isfinite(points).all())}")
    print(f"coordinate bounds (m): {points.min(axis=0).round(3)} to {points.max(axis=0).round(3)}")
    print(f"cloud: {output_path}")
    print(f"plot: {plot_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "capture_root",
        nargs="?",
        type=Path,
        default=Path("data/processed/single_room/c00a170fe1"),
    )
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.02, help="voxel width in metres")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    capture_name = (
        args.capture_root.stem
        if args.capture_root.suffix.lower() == ".zip"
        else args.capture_root.parent.name
    )
    voxel_cm = args.voxel_size * 100
    voxel_label = f"{voxel_cm:g}cm"
    output_path = args.output or Path(
        f"reports/{capture_name}_step{args.frame_step}_voxel{voxel_label}.npz"
    )
    plot_path = args.plot or Path(
        f"reports/{capture_name}_step{args.frame_step}_voxel{voxel_label}.png"
    )
    process_capture(
        args.capture_root,
        output_path,
        plot_path,
        args.frame_step,
        args.voxel_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
