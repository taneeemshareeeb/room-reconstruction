#!/usr/bin/env python3
"""Highlight wall traces supported by more than one height band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from room_reconstruction.consensus import detect_wall_candidates_by_height, find_consensus_walls
from room_reconstruction.geometry import estimate_floor_height
from room_reconstruction.pointcloud import load_archive_point_cloud


DEFAULT_CAPTURE = Path("data/raw/single_scan_with_ceiling.zip")
DEFAULT_IMAGE = Path("reports/single_scan_with_ceiling_wall_consensus.png")
DEFAULT_JSON = Path("reports/single_scan_with_ceiling_wall_consensus.json")
HEIGHT_BANDS = (
    ("lower", 0.40, 0.80, "#1f77b4"),
    ("current", 0.85, 1.55, "#ff7f0e"),
    ("upper", 1.50, 2.20, "#9467bd"),
)


def compare_consensus(
    points: np.ndarray,
    image_path: Path,
    *,
    max_angle_deg: float = 10.0,
    max_offset_m: float = 0.20,
    min_overlap_m: float = 0.75,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must be a non-empty N x 3 array")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates")
    floor_y = estimate_floor_height(points)
    candidates_by_band = detect_wall_candidates_by_height(points)
    consensus = find_consensus_walls(
        candidates_by_band,
        max_angle_deg=max_angle_deg,
        max_offset_m=max_offset_m,
        min_overlap_m=min_overlap_m,
    )

    x_limits = (float(points[:, 0].min()), float(points[:, 0].max()))
    z_limits = (float(points[:, 2].min()), float(points[:, 2].max()))
    combined = points[
        (points[:, 1] >= floor_y + 0.40) & (points[:, 1] <= floor_y + 2.20)
    ]
    display = combined[::max(1, len(combined) // 120_000)]
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True,
                                constrained_layout=True)
    for axis in axes:
        axis.scatter(display[:, 0], display[:, 2], s=0.25, color="#87909a",
                     alpha=0.35, linewidths=0, rasterized=True)
        axis.set_xlim(x_limits)
        axis.set_ylim(z_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("World x (m)")
        axis.grid(alpha=0.15, linewidth=0.5)

    for label, _, _, color in HEIGHT_BANDS:
        for wall in candidates_by_band[label]:
            start, end = wall["start_xz_m"], wall["end_xz_m"]
            axes[0].plot([start[0], end[0]], [start[1], end[1]], color=color,
                         linewidth=1.6, alpha=0.8)
    axes[0].set_title(
        "Detected wall candidates by band\n"
        + " · ".join(f"{label}: {len(candidates_by_band[label])}"
                     for label, *_ in HEIGHT_BANDS)
    )
    axes[1].set_ylabel("World z (m)")
    axes[0].set_ylabel("World z (m)")
    for wall in consensus:
        start, end = wall["start_xz_m"], wall["end_xz_m"]
        axes[1].plot([start[0], end[0]], [start[1], end[1]], color="#18864b",
                     linewidth=2.4, solid_capstyle="round")
        midpoint = (np.asarray(start) + np.asarray(end)) / 2
        axes[1].text(midpoint[0], midpoint[1], str(wall["support_count"]),
                     color="#075b31", fontsize=8, ha="center", va="center",
                     bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    axes[1].set_title(
        f"Cross-band agreements · {len(consensus)} segments\n"
        "Green segments have matching candidates in 2+ bands"
    )
    figure.suptitle(
        f"Wall candidate agreement · angle ≤ {max_angle_deg:g}° · "
        f"offset ≤ {max_offset_m:g} m · overlap ≥ {min_overlap_m:g} m\n"
        "Agreement is a consistency hint, not proof of a real wall"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return candidates_by_band, consensus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--input", type=Path, help="use a saved point-cloud .npz")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-angle-deg", type=float, default=10.0)
    parser.add_argument("--max-offset-m", type=float, default=0.20)
    parser.add_argument("--min-overlap-m", type=float, default=0.75)
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

    candidates, consensus = compare_consensus(
        points, args.image,
        max_angle_deg=args.max_angle_deg,
        max_offset_m=args.max_offset_m,
        min_overlap_m=args.min_overlap_m,
    )
    document = {
        "status": "candidate_unverified",
        "source_capture": str(args.input or args.capture),
        "matching": {
            "max_angle_deg": args.max_angle_deg,
            "max_offset_m": args.max_offset_m,
            "min_overlap_m": args.min_overlap_m,
        },
        "wall_candidates_by_band": candidates,
        "cross_band_consensus_candidates": consensus,
        "note": "Cross-band geometric agreement is not ground-truth validation.",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("wall candidates:")
    for label, *_ in HEIGHT_BANDS:
        print(f"  {label}: {len(candidates[label])}")
    support_counts = {}
    for candidate in consensus:
        support_counts[candidate["support_count"]] = support_counts.get(candidate["support_count"], 0) + 1
    print(f"cross-band consensus segments: {len(consensus)}")
    for bands, count in sorted(support_counts.items()):
        print(f"  supported by {bands} bands: {count}")
    print(f"image: {args.image}")
    print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
