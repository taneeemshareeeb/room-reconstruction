"""Utilities for combining sampled depth frames into a sparse point cloud."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from posixpath import dirname

import numpy as np
from PIL import Image

from .depth import (
    _depth_arrays_to_world_points,
    _load_depth_frame_from_row,
    _validate_frame_options,
)


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.02) -> np.ndarray:
    """Keep the first point in each voxel, preserving its input order."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape N x 3")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and greater than zero")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite coordinates")
    if len(points) == 0:
        return points.copy()

    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    _, first_indices = np.unique(voxel_keys, axis=0, return_index=True)
    return points[np.sort(first_indices)]


def load_capture_point_cloud(
    capture_root: str | Path,
    *,
    frame_step: int = 5,
    frame_range: tuple[int, int] | None = None,
    min_confidence: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    rgb_size: tuple[int, int] = (1920, 1440),
    voxel_size: float = 0.02,
    chunk_frames: int = 20,
) -> np.ndarray:
    """Load sampled odometry frames and return one voxel-downsampled cloud.

    ``frame_step`` and the optional half-open ``frame_range`` select positions
    in the odometry CSV, not numeric frame IDs. PNGs are loaded one frame at a
    time; the combined cloud is voxel-downsampled before it is returned.
    """
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and greater than zero")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be at least 1")
    _validate_frame_options(0, min_depth_m, max_depth_m, rgb_size)

    root = Path(capture_root)
    with (root / "odometry.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    if not rows:
        raise ValueError(f"No odometry rows found in {root / 'odometry.csv'}")

    start, stop = (0, len(rows)) if frame_range is None else frame_range
    if start < 0 or stop < start or stop > len(rows):
        raise ValueError(
            f"frame_range must satisfy 0 <= start <= stop <= {len(rows)}"
        )
    selected_rows = rows[start:stop:frame_step]

    def load_row(row: dict[str, str]) -> np.ndarray:
        return _load_depth_frame_from_row(
            root,
            int(row["frame"]),
            row,
            min_confidence=min_confidence,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            rgb_size=rgb_size,
        )

    return _downsample_rows(selected_rows, load_row, voxel_size, chunk_frames)


def load_archive_point_cloud(
    archive_path: str | Path,
    *,
    frame_step: int = 5,
    frame_range: tuple[int, int] | None = None,
    min_confidence: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    rgb_size: tuple[int, int] = (1920, 1440),
    voxel_size: float = 0.02,
    chunk_frames: int = 20,
) -> np.ndarray:
    """Build a point cloud by reading a capture ZIP directly, without extracting it."""
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and greater than zero")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be at least 1")
    _validate_frame_options(0, min_depth_m, max_depth_m, rgb_size)

    with zipfile.ZipFile(archive_path) as archive:
        odometry_paths = [
            item.filename
            for item in archive.infolist()
            if item.filename.endswith("/odometry.csv")
        ]
        if len(odometry_paths) != 1:
            raise ValueError(
                f"Expected one capture odometry.csv in {archive_path}; "
                f"found {len(odometry_paths)}"
            )
        odometry_path = odometry_paths[0]
        capture_prefix = dirname(odometry_path)
        with archive.open(odometry_path) as binary_stream:
            with io.TextIOWrapper(
                binary_stream, encoding="utf-8-sig", newline=""
            ) as text_stream:
                rows = list(csv.DictReader(text_stream, skipinitialspace=True))
        if not rows:
            raise ValueError(f"No odometry rows found in {archive_path}")

        start, stop = (0, len(rows)) if frame_range is None else frame_range
        if start < 0 or stop < start or stop > len(rows):
            raise ValueError(
                f"frame_range must satisfy 0 <= start <= stop <= {len(rows)}"
            )
        selected_rows = rows[start:stop:frame_step]

        def load_row(row: dict[str, str]) -> np.ndarray:
            frame_id = int(row["frame"])
            filename = f"{frame_id:06d}.png"
            with archive.open(f"{capture_prefix}/depth/{filename}") as stream:
                with Image.open(stream) as image:
                    depth = np.asarray(image).copy()
            with archive.open(f"{capture_prefix}/confidence/{filename}") as stream:
                with Image.open(stream) as image:
                    confidence = np.asarray(image).copy()
            return _depth_arrays_to_world_points(
                depth,
                confidence,
                row,
                min_confidence=min_confidence,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                rgb_size=rgb_size,
            )

        return _downsample_rows(selected_rows, load_row, voxel_size, chunk_frames)


def _downsample_rows(
    rows: list[dict[str, str]],
    load_row,
    voxel_size: float,
    chunk_frames: int,
) -> np.ndarray:
    """Reduce memory by voxelizing frame batches before the final merge."""
    if not rows:
        return np.empty((0, 3), dtype=np.float64)

    reduced_chunks = []
    for offset in range(0, len(rows), chunk_frames):
        frame_clouds = [load_row(row) for row in rows[offset : offset + chunk_frames]]
        batch = np.concatenate(frame_clouds, axis=0)
        reduced_chunks.append(voxel_downsample(batch, voxel_size=voxel_size))
    merged = np.concatenate(reduced_chunks, axis=0)
    return voxel_downsample(merged, voxel_size=voxel_size)
