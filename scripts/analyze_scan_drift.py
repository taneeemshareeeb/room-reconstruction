#!/usr/bin/env python3
"""Measure repeat-pass consistency in a ceiling-inclusive LiDAR capture.

The input ZIP is read directly. No files are extracted or changed. The report
compares pose paths and wall-slice surfaces, and labels all agreement measures
as consistency rather than accuracy or drift ground truth.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from room_reconstruction.geometry import estimate_floor_height
from room_reconstruction.pointcloud import load_archive_point_cloud


def _read_poses(archive_path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    with zipfile.ZipFile(archive_path) as archive:
        paths = [
            item.filename for item in archive.infolist()
            if Path(item.filename).name.lower() == "odometry.csv"
        ]
        if len(paths) != 1:
            raise ValueError(f"expected one odometry.csv, found {len(paths)}")
        with archive.open(paths[0]) as stream:
            text = stream.read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text), skipinitialspace=True))
    if not rows:
        raise ValueError("odometry.csv contains no pose rows")
    required = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"odometry.csv is missing columns: {', '.join(missing)}")
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    positions = np.asarray(
        [[float(row[axis]) for axis in ("x", "y", "z")] for row in rows],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(timestamps).all() or not np.isfinite(positions).all():
        raise ValueError("timestamps and positions must be finite")
    if not np.isfinite(quaternions).all() or np.any(np.linalg.norm(quaternions, axis=1) < 1e-12):
        raise ValueError("pose quaternions must be finite and nonzero")
    return rows, timestamps, positions


def _detect_pitch_split(rows: list[dict[str, str]], timestamps: np.ndarray) -> tuple[int, dict[str, float]]:
    """Find a two-regime split in smoothed camera view-direction y values."""
    quaternions = np.asarray(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows],
        dtype=np.float64,
    )
    view_y = Rotation.from_quat(quaternions).as_matrix()[:, 1, 2]
    intervals = np.diff(timestamps)
    if len(intervals) == 0 or not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError("timestamps must increase strictly to detect scan passes")
    frame_rate = 1.0 / float(np.median(intervals))
    window = max(3, int(round(frame_rate)))
    smoothed = np.convolve(view_y, np.ones(window) / window, mode="same")

    # Find the single change point that best separates the two pitch regimes.
    lower, upper = max(1, len(smoothed) // 5), min(len(smoothed) - 1, 4 * len(smoothed) // 5)
    prefix = np.concatenate(([0.0], np.cumsum(smoothed)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(smoothed * smoothed)))
    total_sq = prefix_sq[-1]
    best_index, best_error = None, math.inf
    for index in range(lower, upper + 1):
        left_n, right_n = index, len(smoothed) - index
        left_sum, right_sum = prefix[index], prefix[-1] - prefix[index]
        error = (
            prefix_sq[index] - left_sum * left_sum / left_n
            + total_sq - prefix_sq[index] - right_sum * right_sum / right_n
        )
        if error < best_error:
            best_index, best_error = index, error
    if best_index is None:
        raise ValueError("not enough poses to find a pass boundary")
    means = (float(smoothed[:best_index].mean()), float(smoothed[best_index:].mean()))
    if means[0] >= -0.1 or means[1] <= 0.1:
        raise ValueError(
            "camera pitch does not separate into a downward-looking pass and an upward-looking pass"
        )
    return best_index, {
        "frame_rate_hz": frame_rate,
        "smoothing_window_frames": window,
        "mean_view_y_pass_a": means[0],
        "mean_view_y_pass_b": means[1],
        "smoothed_view_y_at_split": float(smoothed[best_index]),
    }


def _route_summary(positions: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    return {
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "path_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "start_to_end_gap_m": float(np.linalg.norm(positions[-1] - positions[0])),
    }


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return cKDTree(target).query(source, workers=-1)[0]


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    return {
        f"p{percentile}_m": float(np.percentile(values, percentile))
        for percentile in (50, 90, 95)
    } | {
        "within_0_10m_fraction": float(np.mean(values <= 0.10)),
        "within_0_20m_fraction": float(np.mean(values <= 0.20)),
    }


def _rigid_fit_once(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one trimmed 2D rigid transform from nearest wall points."""
    distances, indices = cKDTree(target).query(source, workers=-1)
    keep = distances <= min(0.35, float(np.percentile(distances, 80)))
    if int(keep.sum()) < 100:
        raise ValueError("too few nearby wall points to estimate a rigid pass alignment")
    src = source[keep]
    dst = target[indices[keep]]
    src_center, dst_center = src.mean(axis=0), dst.mean(axis=0)
    u, _, vh = np.linalg.svd((src - src_center).T @ (dst - dst_center))
    rotation = u @ np.diag([1.0, np.linalg.det(u @ vh)]) @ vh
    translation = dst_center - src_center @ rotation
    return rotation, translation


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def analyze(archive_path: Path, *, frame_step: int, voxel_size_m: float) -> tuple[dict[str, object], dict[str, object]]:
    rows, timestamps, positions = _read_poses(archive_path)
    split, split_method = _detect_pitch_split(rows, timestamps)
    pass_a = positions[:split]
    pass_b = positions[split:]
    pose_a = _route_summary(pass_a, timestamps[:split])
    pose_b = _route_summary(pass_b, timestamps[split:])

    path_a = pass_a[::max(1, len(pass_a) // 2_000), :][:, (0, 2)]
    path_b = pass_b[::max(1, len(pass_b) // 2_000), :][:, (0, 2)]
    trajectory_a_to_b = _nearest_distances(path_a, path_b)
    trajectory_b_to_a = _nearest_distances(path_b, path_a)

    cloud_a = load_archive_point_cloud(
        archive_path, frame_step=frame_step, frame_range=(0, split), voxel_size=voxel_size_m
    )
    cloud_b = load_archive_point_cloud(
        archive_path, frame_step=frame_step, frame_range=(split, len(rows)), voxel_size=voxel_size_m
    )
    floor_a, floor_b = estimate_floor_height(cloud_a), estimate_floor_height(cloud_b)
    wall_a = cloud_a[
        (cloud_a[:, 1] >= floor_a + 0.3) & (cloud_a[:, 1] <= floor_a + 1.8)
    ][:, (0, 2)]
    wall_b = cloud_b[
        (cloud_b[:, 1] >= floor_b + 0.3) & (cloud_b[:, 1] <= floor_b + 1.8)
    ][:, (0, 2)]
    if min(len(wall_a), len(wall_b)) < 100:
        raise ValueError("too few points in one pass's wall-height slice for comparison")

    # Keep ICP deterministic and bounded while retaining a spatially distributed sample.
    sample_a = wall_a[::max(1, len(wall_a) // 30_000)]
    sample_b = wall_b[::max(1, len(wall_b) // 30_000)]
    raw_b_to_a = _nearest_distances(sample_b, sample_a)
    raw_a_to_b = _nearest_distances(sample_a, sample_b)
    rotation, translation = _rigid_fit_once(sample_b, sample_a)
    aligned_b = sample_b @ rotation + translation
    aligned_a_to_b = _nearest_distances(sample_a, aligned_b)
    aligned_b_to_a = _nearest_distances(aligned_b, sample_a)

    transform_angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    report: dict[str, object] = {
        "capture": archive_path.name,
        "method": {
            "pass_split": "one-second-smoothed camera view direction; minimum two-mean change-point error",
            "wall_slice_m_above_each_pass_floor": [0.3, 1.8],
            "rigid_alignment": "one trimmed 2D Procrustes step on nearest wall-slice points",
            "frame_step": frame_step,
            "voxel_size_m": voxel_size_m,
        },
        "pass_split": {
            "row_index": split,
            "elapsed_time_s": float(timestamps[split] - timestamps[0]),
            "pass_a_frames": split,
            "pass_b_frames": len(rows) - split,
            **split_method,
        },
        "pose_path": {
            "pass_a": pose_a,
            "pass_b": pose_b,
            "repeat_path_nearest_neighbor_xz_m": {
                "pass_a_to_b": _distance_summary(trajectory_a_to_b),
                "pass_b_to_a": _distance_summary(trajectory_b_to_a),
            },
        },
        "wall_surface_consistency": {
            "pass_a_wall_points": len(wall_a),
            "pass_b_wall_points": len(wall_b),
            "floor_height_estimates_m": [float(floor_a), float(floor_b)],
            "nearest_wall_point_distance_m_before": {
                "pass_b_to_a": _distance_summary(raw_b_to_a),
                "pass_a_to_b": _distance_summary(raw_a_to_b),
            },
            "one_step_global_rigid_alignment": {
                "rotation_deg": transform_angle,
                "translation_xz_m": translation.tolist(),
                "nearest_wall_point_distance_m_after": {
                    "pass_b_to_a": _distance_summary(aligned_b_to_a),
                    "pass_a_to_b": _distance_summary(aligned_a_to_b),
                },
            },
        },
        "interpretation": (
            "These are repeatability and overlap measurements, not accuracy or calibrated drift. "
            "A single rigid pass transform cannot correct position-dependent drift; compare its "
            "before/after overlap before considering it for reconstruction."
        ),
    }
    plot_data = {
        "positions_a": pass_a[:, (0, 2)],
        "positions_b": pass_b[:, (0, 2)],
        "wall_a": sample_a,
        "wall_b": sample_b,
        "wall_b_aligned": aligned_b,
        "wall_a_to_b_before": raw_a_to_b,
        "wall_b_to_a_before": raw_b_to_a,
        "wall_a_to_b_after": aligned_a_to_b,
        "wall_b_to_a_after": aligned_b_to_a,
    }
    return report, plot_data


def render_report(report: dict[str, object], plot_data: dict[str, object], output_path: Path) -> None:
    figure, (trajectory_axis, distance_axis) = plt.subplots(1, 2, figsize=(13, 6))
    a = plot_data["positions_a"]
    b = plot_data["positions_b"]
    trajectory_axis.plot(a[:, 0], a[:, 1], color="#2468a2", label="Pass A · walls/floor")
    trajectory_axis.plot(b[:, 0], b[:, 1], color="#d97924", label="Pass B · ceiling")
    trajectory_axis.scatter([a[0, 0]], [a[0, 1]], color="#2468a2", marker="o", zorder=3)
    trajectory_axis.scatter([b[0, 0]], [b[0, 1]], color="#d97924", marker="s", zorder=3)
    trajectory_axis.set_title("Recorded camera paths")
    trajectory_axis.set_xlabel("World x (m)")
    trajectory_axis.set_ylabel("World z (m)")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.grid(alpha=0.2)
    trajectory_axis.legend(loc="best", fontsize=8)

    for values, label, color, style in (
        (plot_data["wall_b_to_a_before"], "B → A, as recorded", "#d97924", "-"),
        (plot_data["wall_a_to_b_before"], "A → B, as recorded", "#2468a2", "-"),
        (plot_data["wall_b_to_a_after"], "B → A, one rigid step", "#d97924", "--"),
        (plot_data["wall_a_to_b_after"], "A → B, one rigid step", "#2468a2", "--"),
    ):
        x, y = _empirical_cdf(values)
        distance_axis.plot(x, y, color=color, linestyle=style, label=label)
    distance_axis.set_xlim(0, 0.6)
    distance_axis.set_ylim(0, 1)
    distance_axis.set_title("Wall-slice nearest-point overlap")
    distance_axis.set_xlabel("Distance to other pass (m)")
    distance_axis.set_ylabel("Fraction of points")
    distance_axis.grid(alpha=0.2)
    distance_axis.legend(loc="lower right", fontsize=8)
    split = report["pass_split"]
    figure.suptitle(
        f"{report['capture']} · pitch split at {split['elapsed_time_s']:.1f} s\n"
        "CONSISTENCY CHECK — NOT GROUND-TRUTH DRIFT",
        fontsize=13,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare repeated pose paths and wall-surface overlap without extracting the capture ZIP."
    )
    parser.add_argument("capture", type=Path, nargs="?", default=Path("data/raw/single_scan_with_ceiling.zip"))
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/drift_analysis"))
    args = parser.parse_args(argv)
    if args.frame_step < 1 or not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("frame-step and voxel-size must be positive")
    try:
        report, plot_data = analyze(
            args.capture, frame_step=args.frame_step, voxel_size_m=args.voxel_size
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "drift_analysis.json"
        image_path = args.output_dir / "drift_analysis.png"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        render_report(report, plot_data, image_path)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Could not analyze scan passes: {error}", file=sys.stderr)
        return 1

    split = report["pass_split"]
    pose = report["pose_path"]
    surfaces = report["wall_surface_consistency"]
    before = surfaces["nearest_wall_point_distance_m_before"]
    after = surfaces["one_step_global_rigid_alignment"]["nearest_wall_point_distance_m_after"]
    print(f"Pass split: row {split['row_index']} at {split['elapsed_time_s']:.2f} s")
    print(
        "Path length / endpoint gap: "
        f"A {pose['pass_a']['path_length_m']:.2f} / {pose['pass_a']['start_to_end_gap_m']:.2f} m, "
        f"B {pose['pass_b']['path_length_m']:.2f} / {pose['pass_b']['start_to_end_gap_m']:.2f} m"
    )
    print(
        "Wall overlap p95 B→A, A→B: "
        f"{before['pass_b_to_a']['p95_m']:.3f}, {before['pass_a_to_b']['p95_m']:.3f} m before; "
        f"{after['pass_b_to_a']['p95_m']:.3f}, {after['pass_a_to_b']['p95_m']:.3f} m after one rigid step"
    )
    print(f"Report: {json_path}\nPreview: {image_path}")
    print("Consistency only; the global alignment is diagnostic, not a drift correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
