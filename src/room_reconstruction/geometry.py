"""Conservative geometry candidates derived from a world-space point cloud."""

from __future__ import annotations

import numpy as np


def estimate_floor_height(points: np.ndarray) -> float:
    """Estimate floor elevation from the lowest strong mode of world y."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
        raise ValueError("points must be a non-empty N x 3 array")
    heights, edges = np.histogram(
        cloud[:, 1], bins=max(32, int(np.ptp(cloud[:, 1]) / 0.03))
    )
    strong_modes = np.flatnonzero(heights >= heights.max() * 0.35)
    peak = int(strong_modes[0])
    return float((edges[peak] + edges[peak + 1]) / 2)


def detect_wall_candidates(
    points: np.ndarray,
    *,
    slice_min_height_m: float = 0.85,
    slice_max_height_m: float = 1.55,
    grid_size_m: float = 0.08,
    line_tolerance_m: float = 0.10,
    min_length_m: float = 1.5,
    max_lines: int = 24,
    random_seed: int = 17,
) -> list[dict[str, object]]:
    """Find long line traces in a horizontal slice above the estimated floor.

    The densest low-height band estimates floor elevation. Points between the
    requested heights above it are projected to x/z before fitting. The result
    is candidates only: it does not claim that a trace is a verified wall or
    assign walls to rooms. This assumes world y is up and the low mode is floor.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("points must have shape N x 3")
    if not np.isfinite(cloud).all():
        raise ValueError("points must contain only finite coordinates")
    if (
        not np.isfinite(slice_min_height_m)
        or not np.isfinite(slice_max_height_m)
        or slice_min_height_m < 0
        or slice_max_height_m <= slice_min_height_m
    ):
        raise ValueError("slice heights must be finite, non-negative, and ordered")
    if grid_size_m <= 0 or line_tolerance_m <= 0:
        raise ValueError("grid size and line tolerance must be positive")
    if min_length_m <= 0 or max_lines < 1:
        raise ValueError("minimum length and maximum lines must be positive")
    if len(cloud) < 2:
        return []

    # A histogram's lowest strong mode is a practical floor estimate for these
    # captures. A narrow slice avoids mixing floor/ceiling passes into the plan.
    floor_y = estimate_floor_height(cloud)
    slice_points = cloud[
        (cloud[:, 1] >= floor_y + slice_min_height_m)
        & (cloud[:, 1] <= floor_y + slice_max_height_m)
    ]
    if len(slice_points) < 2:
        return []
    cell_ids = np.floor(slice_points[:, (0, 2)] / grid_size_m).astype(np.int64)
    cells = np.unique(cell_ids, axis=0)
    horizontal = (cells.astype(np.float64) + 0.5) * grid_size_m
    if len(horizontal) < 2:
        return []

    rng = np.random.default_rng(random_seed)
    remaining = np.arange(len(horizontal))
    raw_lines: list[tuple[np.ndarray, np.ndarray, int]] = []
    iterations = min(700, max(160, len(horizontal) // 12))

    for _ in range(max_lines * 2):
        if len(remaining) < 2:
            break
        best_inliers = np.empty(0, dtype=np.int64)
        for _ in range(iterations):
            first, second = rng.choice(remaining, size=2, replace=False)
            delta = horizontal[second] - horizontal[first]
            norm = np.linalg.norm(delta)
            if norm < min_length_m * 0.4:
                continue
            direction = delta / norm
            normal = np.array([-direction[1], direction[0]])
            residual = np.abs((horizontal[remaining] - horizontal[first]) @ normal)
            inliers = np.flatnonzero(residual <= line_tolerance_m)
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        if len(best_inliers) < 12:
            break

        chosen_global = remaining[best_inliers]
        sample = horizontal[chosen_global]
        center = sample.mean(axis=0)
        _, _, vectors = np.linalg.svd(sample - center, full_matrices=False)
        direction = vectors[0]
        projection = (sample - center) @ direction
        order = np.argsort(projection)
        sorted_projection = projection[order]
        split_at = np.flatnonzero(np.diff(sorted_projection) > 0.35) + 1
        for group in np.split(order, split_at):
            if len(group) < 8:
                continue
            segment = sample[group]
            segment_center = segment.mean(axis=0)
            _, _, segment_vectors = np.linalg.svd(
                segment - segment_center, full_matrices=False
            )
            segment_direction = segment_vectors[0]
            segment_projection = (segment - segment_center) @ segment_direction
            start_t = float(segment_projection.min())
            end_t = float(segment_projection.max())
            if end_t - start_t < min_length_m:
                continue
            start = segment_center + start_t * segment_direction
            end = segment_center + end_t * segment_direction
            raw_lines.append((start, end, len(group)))
        remaining = np.delete(remaining, best_inliers)
        if len(raw_lines) >= max_lines * 2:
            break

    # Merge near-duplicate traces caused by the thickness of a wall surface.
    merged: list[dict[str, object]] = []
    for start, end, support in sorted(raw_lines, key=lambda item: item[2], reverse=True):
        delta = end - start
        direction = delta / np.linalg.norm(delta)
        if direction[0] < 0 or (abs(direction[0]) < 1e-8 and direction[1] < 0):
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        offset = float(np.dot((start + end) / 2, normal))
        match = None
        for line in merged:
            dot = float(np.clip(np.dot(direction, line["direction"]), -1, 1))
            angle = np.arccos(abs(dot))
            if angle <= np.deg2rad(8) and abs(offset - line["offset"]) <= 0.45:
                match = line
                break
        if match is None:
            merged.append({"direction": direction, "normal": normal,
                           "offset": offset, "segments": [(start, end, support)]})
        else:
            match["segments"].append((start, end, support))

    result: list[dict[str, object]] = []
    for line in merged:
        direction = line["direction"]
        intervals = []
        for start, end, support in line["segments"]:
            a, b = sorted((float(np.dot(start, direction)), float(np.dot(end, direction))))
            intervals.append((a, b, support))
        intervals.sort(key=lambda interval: interval[0])
        groups: list[list[tuple[float, float, int]]] = []
        for interval in intervals:
            if not groups or interval[0] - max(item[1] for item in groups[-1]) > 0.30:
                groups.append([interval])
            else:
                groups[-1].append(interval)

        for group in groups:
            start_t = min(item[0] for item in group)
            end_t = max(item[1] for item in group)
            length = end_t - start_t
            support_cells = int(sum(item[2] for item in group))
            if length < min_length_m or support_cells < 20:
                continue
            start = direction * start_t + line["normal"] * line["offset"]
            end = direction * end_t + line["normal"] * line["offset"]
            result.append({
                "id": f"wall_candidate_{len(result) + 1:02d}",
                "start_xz_m": [round(float(v), 3) for v in start],
                "end_xz_m": [round(float(v), 3) for v in end],
                "length_m": round(float(length), 3),
                "supporting_cells": support_cells,
                "status": "candidate_unverified",
                "reason": (
                    f"Line fit in the {floor_y + slice_min_height_m:.2f}–"
                    f"{floor_y + slice_max_height_m:.2f} m "
                    "height slice; visual confirmation required."
                ),
            })
    return result
