"""Detect repeated camera passes from LiDAR capture pose metadata."""

from __future__ import annotations

import csv
import io
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class NoPassSplitError(ValueError):
    """Raised when valid pose data does not show two opposing view passes."""


def _read_odometry(capture: Path) -> list[dict[str, str]]:
    if capture.is_dir():
        with (capture / "odometry.csv").open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
    else:
        with zipfile.ZipFile(capture) as archive:
            paths = [
                item.filename for item in archive.infolist()
                if Path(item.filename).name.lower() == "odometry.csv"
            ]
            if len(paths) != 1:
                raise ValueError(f"expected one odometry.csv, found {len(paths)}")
            with archive.open(paths[0]) as binary_stream:
                text = binary_stream.read().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text), skipinitialspace=True))
    if not rows:
        raise ValueError("odometry.csv contains no pose rows")
    return rows


def detect_camera_pass_split(capture: str | Path) -> dict[str, float | int]:
    """Find a downward-to-upward view change using a one-second pitch smoother.

    The split is a frame-row index into odometry.csv. This is a capture-specific
    pass boundary estimate, not evidence that either path returned to ground
    truth or that their world poses are drift-free.
    """
    rows = _read_odometry(Path(capture))
    required = {"timestamp", "qx", "qy", "qz", "qw"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"odometry.csv is missing columns: {', '.join(missing)}")
    timestamps = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    quaternions = np.asarray(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows],
        dtype=np.float64,
    )
    intervals = np.diff(timestamps)
    if (
        not np.isfinite(timestamps).all()
        or not np.isfinite(quaternions).all()
        or np.any(np.linalg.norm(quaternions, axis=1) < 1e-12)
        or len(intervals) == 0
        or not np.isfinite(intervals).all()
        or np.any(intervals <= 0)
    ):
        raise ValueError("odometry timestamps or quaternions are invalid")
    view_y = Rotation.from_quat(quaternions).as_matrix()[:, 1, 2]
    frame_rate = 1.0 / float(np.median(intervals))
    window = max(3, int(round(frame_rate)))
    smoothed = np.convolve(view_y, np.ones(window) / window, mode="same")

    lower = max(1, len(smoothed) // 5)
    upper = min(len(smoothed) - 1, 4 * len(smoothed) // 5)
    prefix = np.concatenate(([0.0], np.cumsum(smoothed)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(smoothed * smoothed)))
    total_sq = prefix_sq[-1]
    best_index, best_error = None, math.inf
    for index in range(lower, upper + 1):
        left_n, right_n = index, len(smoothed) - index
        left_sum = prefix[index]
        right_sum = prefix[-1] - left_sum
        error = (
            prefix_sq[index] - left_sum * left_sum / left_n
            + total_sq - prefix_sq[index] - right_sum * right_sum / right_n
        )
        if error < best_error:
            best_index, best_error = index, error
    if best_index is None:
        raise ValueError("not enough poses to detect a pass split")
    mean_a = float(smoothed[:best_index].mean())
    mean_b = float(smoothed[best_index:].mean())
    if mean_a >= -0.1 or mean_b <= 0.1:
        raise NoPassSplitError(
            "camera pitch does not show a down-looking pass followed by an up-looking pass"
        )
    return {
        "split_row_index": best_index,
        "frame_count": len(rows),
        "elapsed_time_s": float(timestamps[best_index] - timestamps[0]),
        "pass_a_view_y_mean": mean_a,
        "pass_b_view_y_mean": mean_b,
        "smoothing_window_frames": window,
    }
