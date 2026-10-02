"""Load one depth frame and transform its valid pixels into world coordinates."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def _odometry_row(capture_root: Path, frame_id: int) -> dict[str, str]:
    path = capture_root / "odometry.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"Odometry CSV has no header: {path}")
        matches = [row for row in reader if int(row["frame"]) == frame_id]
    if not matches:
        raise ValueError(f"Frame {frame_id:06d} is not present in {path}")
    if len(matches) > 1:
        raise ValueError(f"Frame {frame_id:06d} appears more than once in {path}")
    return matches[0]


def _validate_frame_options(
    frame_id: int,
    min_depth_m: float,
    max_depth_m: float,
    rgb_size: tuple[int, int],
) -> None:
    if frame_id < 0:
        raise ValueError("frame_id must be zero or greater")
    if (
        not np.isfinite([min_depth_m, max_depth_m]).all()
        or min_depth_m < 0
        or max_depth_m <= min_depth_m
    ):
        raise ValueError("depth bounds must satisfy 0 <= min_depth_m < max_depth_m")
    if len(rgb_size) != 2 or min(rgb_size) <= 0:
        raise ValueError("rgb_size must contain positive (width, height) values")


def load_depth_frame(
    capture_root: str | Path,
    frame_id: int,
    *,
    min_confidence: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    rgb_size: tuple[int, int] = (1920, 1440),
) -> np.ndarray:
    """Return valid points from one frame as an ``N x 3`` world-coordinate array.

    Depth PNG values are interpreted as millimetres. Camera coordinates follow
    the OpenCV convention (x right, y down, z forward); the pose quaternion and
    translation map camera points into world coordinates as ``cam @ R.T + t``.

    ``rgb_size`` is ``(width, height)`` for the image whose intrinsics are
    recorded in ``odometry.csv``.
    """
    root = Path(capture_root)
    _validate_frame_options(frame_id, min_depth_m, max_depth_m, rgb_size)

    row = _odometry_row(root, frame_id)
    return _load_depth_frame_from_row(
        root,
        frame_id,
        row,
        min_confidence=min_confidence,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        rgb_size=rgb_size,
    )


def _load_depth_frame_from_row(
    root: Path,
    frame_id: int,
    row: dict[str, str],
    *,
    min_confidence: int,
    min_depth_m: float,
    max_depth_m: float,
    rgb_size: tuple[int, int],
) -> np.ndarray:
    frame_name = f"{frame_id:06d}.png"
    with Image.open(root / "depth" / frame_name) as image:
        depth_raw = np.asarray(image).copy()
    with Image.open(root / "confidence" / frame_name) as image:
        confidence = np.asarray(image).copy()

    return _depth_arrays_to_world_points(
        depth_raw,
        confidence,
        row,
        min_confidence=min_confidence,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        rgb_size=rgb_size,
    )


def _depth_arrays_to_world_points(
    depth_raw: np.ndarray,
    confidence: np.ndarray,
    row: dict[str, str],
    *,
    min_confidence: int,
    min_depth_m: float,
    max_depth_m: float,
    rgb_size: tuple[int, int],
) -> np.ndarray:

    if depth_raw.ndim != 2 or confidence.ndim != 2:
        raise ValueError("Depth and confidence images must be single-channel images")
    if depth_raw.shape != confidence.shape:
        raise ValueError(
            f"Depth/confidence image dimensions differ: {depth_raw.shape} vs {confidence.shape}"
        )
    if depth_raw.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth in millimetres, got {depth_raw.dtype}")
    if confidence.dtype != np.uint8:
        raise ValueError(f"Expected uint8 confidence values, got {confidence.dtype}")
    confidence_values = np.unique(confidence)
    if not np.isin(confidence_values, (0, 1, 2)).all():
        raise ValueError(
            f"Expected confidence values 0, 1, or 2; got {confidence_values.tolist()}"
        )

    depth_m = depth_raw.astype(np.float32) / 1000.0
    height, width = depth_m.shape
    scale_x = width / rgb_size[0]
    scale_y = height / rgb_size[1]
    fx = float(row["fx"]) * scale_x
    fy = float(row["fy"]) * scale_y
    cx = float(row["cx"]) * scale_x
    cy = float(row["cy"]) * scale_y
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")

    valid = (
        (confidence >= min_confidence)
        & (depth_m > min_depth_m)
        & (depth_m < max_depth_m)
    )
    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    camera_points = np.column_stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z)
    )

    quaternion = np.array(
        [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
        dtype=np.float64,
    )
    translation = np.array(
        [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
    )
    if not np.isfinite(quaternion).all() or not np.isfinite(translation).all():
        raise ValueError("Pose quaternion and translation must be finite")
    rotation = Rotation.from_quat(quaternion).as_matrix()
    world_points = camera_points @ rotation.T + translation
    if not np.isfinite(world_points).all():
        raise ValueError("Point-cloud transform produced non-finite coordinates")
    return world_points
