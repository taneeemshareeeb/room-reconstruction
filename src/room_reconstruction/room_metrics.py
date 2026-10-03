"""Attach local footprint and pass-specific surface evidence to room regions."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


def _points_inside_polygon(points_xz: np.ndarray, polygon_xz: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting test for points within a simple polygon."""
    x = points_xz[:, 0]
    z = points_xz[:, 1]
    inside = np.zeros(len(points_xz), dtype=bool)
    previous = polygon_xz[-1]
    for current in polygon_xz:
        x0, z0 = previous
        x1, z1 = current
        crosses = (z0 > z) != (z1 > z)
        crossing_x = (x1 - x0) * (z - z0) / (z1 - z0 + np.finfo(float).eps) + x0
        inside ^= crosses & (x < crossing_x)
        previous = current
    return inside


def _distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    ring = polygon[:-1] if np.allclose(polygon[0], polygon[-1]) else polygon
    starts = ring
    ends = np.roll(ring, -1, axis=0)
    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors),
        lengths_sq,
        out=np.zeros_like(lengths_sq),
        where=lengths_sq > 0,
    )
    projections = starts + np.clip(fractions, 0, 1)[:, None] * vectors
    return float(np.linalg.norm(projections - point, axis=1).min())


def _polygon_metrics(polygon: np.ndarray, axis_deg: float) -> dict[str, object]:
    open_polygon = polygon[:-1] if np.allclose(polygon[0], polygon[-1]) else polygon
    following = np.roll(open_polygon, -1, axis=0)
    area = abs(float(
        np.dot(open_polygon[:, 0], following[:, 1])
        - np.dot(open_polygon[:, 1], following[:, 0])
    )) / 2
    perimeter = float(np.linalg.norm(following - open_polygon, axis=1).sum())
    angle = math.radians(axis_deg)
    basis = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    local = open_polygon @ basis
    spans = np.ptp(local, axis=0)
    return {
        "polygon_area_m2": round(area, 2),
        "polygon_perimeter_m": round(perimeter, 2),
        "dominant_axis_spans_m": [round(float(value), 2) for value in spans],
        "dominant_axis_rotation_deg": round(float(axis_deg), 2),
        "status": "candidate_unverified",
        "reason": "Derived from a height-slice occupancy outline; gaps, drift, and incomplete boundaries can bias these values.",
    }


def _floor_surface(points: np.ndarray, polygon: np.ndarray, global_floor_y: float) -> dict[str, object]:
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    nearby = points[
        (points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
        & (points[:, 2] >= low[1]) & (points[:, 2] <= high[1])
    ]
    if len(nearby):
        nearby = nearby[_points_inside_polygon(nearby[:, (0, 2)], polygon)]
    floor_points = nearby[
        np.abs(nearby[:, 1] - global_floor_y) <= 0.20
    ] if len(nearby) else nearby
    if len(floor_points) < 20:
        return {
            "elevation_world_y_m": None,
            "height_above_global_floor_m": None,
            "supporting_points": int(len(floor_points)),
            "observed_p05_p95_span_m": None,
            "status": "insufficient_data",
            "reason": "Fewer than 20 downward-pass points support the global floor mode inside this candidate region.",
        }
    elevation = float(np.median(floor_points[:, 1]))
    p05, p95 = np.percentile(floor_points[:, 1], [5, 95])
    return {
        "elevation_world_y_m": round(elevation, 3),
        "height_above_global_floor_m": round(elevation - global_floor_y, 3),
        "supporting_points": int(len(floor_points)),
        "observed_p05_p95_span_m": [round(float(p05), 3), round(float(p95), 3)],
        "status": "candidate_unverified",
        "reason": "Median of downward-pass floor-band points within the candidate polygon; interval is observed spread, not calibrated uncertainty.",
    }


def _upper_surfaces(
    points: np.ndarray | None,
    polygon: np.ndarray,
    floor_y: float | None,
    *,
    bin_size_m: float = 0.03,
) -> dict[str, object]:
    if points is None or floor_y is None:
        return {
            "supporting_points_in_height_range": 0,
            "height_candidates": [],
            "status": "insufficient_data",
        }
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    nearby = points[
        (points[:, 0] >= low[0]) & (points[:, 0] <= high[0])
        & (points[:, 2] >= low[1]) & (points[:, 2] <= high[1])
    ]
    if len(nearby):
        nearby = nearby[_points_inside_polygon(nearby[:, (0, 2)], polygon)]
    heights = nearby[
        (nearby[:, 1] >= floor_y + 1.8) & (nearby[:, 1] <= floor_y + 4.2), 1
    ] if len(nearby) else np.empty(0, dtype=np.float64)
    if len(heights) < 30:
        return {
            "supporting_points_in_height_range": int(len(heights)),
            "height_candidates": [],
            "status": "insufficient_data",
        }
    edges = np.arange(floor_y + 1.8, floor_y + 4.2 + bin_size_m, bin_size_m)
    counts, edges = np.histogram(heights, bins=edges)
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=1)
    peaks, properties = find_peaks(
        smoothed,
        prominence=max(3.0, float(smoothed.max()) * 0.05),
        distance=max(1, round(0.25 / bin_size_m)),
    )
    widths, _, _, _ = peak_widths(smoothed, peaks, rel_height=0.5) if len(peaks) else ([], [], [], [])
    ranked = sorted(peaks, key=lambda index: smoothed[index], reverse=True)[:4]
    candidates = []
    for peak in ranked:
        peak_position = int(np.flatnonzero(peaks == peak)[0])
        elevation = float((edges[peak] + edges[peak + 1]) / 2)
        candidates.append({
            "elevation_world_y_m": round(elevation, 3),
            "height_above_local_floor_m": round(elevation - floor_y, 3),
            "supporting_points": int(counts[peak]),
            "observed_half_width_m": round(float(widths[peak_position] * bin_size_m / 2), 3),
            "status": "candidate_unverified",
            "reason": "Upward-pass height peak within this room-region polygon; not proven to be a ceiling surface.",
        })
    return {
        "supporting_points_in_height_range": int(len(heights)),
        "height_candidates": candidates,
        "status": "candidate_available" if candidates else "insufficient_data",
    }


def measure_room_regions(
    room_candidates: list[dict[str, object]],
    floor_pass_points: np.ndarray,
    upper_pass_points: np.ndarray | None,
    global_floor_y: float,
    *,
    dominant_axis_deg: float = 0.0,
) -> list[dict[str, object]]:
    """Add rough geometry and pass-specific surface observations per region."""
    floor_cloud = np.asarray(floor_pass_points, dtype=np.float64)
    upper_cloud = None if upper_pass_points is None else np.asarray(upper_pass_points, dtype=np.float64)
    measured = []
    for candidate in room_candidates:
        polygon = np.asarray(candidate["boundary_xz_m"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4:
            measured.append({
                **candidate,
                "geometry_candidate": None,
                "floor_surface_candidate": None,
                "upper_surface_candidates": {"status": "insufficient_data", "height_candidates": []},
            })
            continue
        floor = _floor_surface(floor_cloud, polygon, global_floor_y)
        local_floor = floor["elevation_world_y_m"]
        measured.append({
            **candidate,
            "geometry_candidate": _polygon_metrics(polygon, dominant_axis_deg),
            "floor_surface_candidate": floor,
            "upper_surface_candidates": _upper_surfaces(
                upper_cloud, polygon,
                float(local_floor) if local_floor is not None else global_floor_y,
            ),
        })
    return measured


def attach_nearby_wall_evidence(
    room_candidates: list[dict[str, object]],
    wall_candidates: list[dict[str, object]],
    wall_gap_candidates: list[dict[str, object]],
    *,
    maximum_boundary_distance_m: float = 0.7,
) -> list[dict[str, object]]:
    """Link nearby scan traces to room regions without asserting they are walls."""
    enriched = []
    for room in room_candidates:
        polygon = np.asarray(room["boundary_xz_m"], dtype=np.float64)
        nearby_walls = []
        for wall in wall_candidates:
            midpoint = (
                np.asarray(wall["start_xz_m"], dtype=np.float64)
                + np.asarray(wall["end_xz_m"], dtype=np.float64)
            ) / 2
            distance = _distance_to_polygon(midpoint, polygon)
            if distance <= maximum_boundary_distance_m:
                nearby_walls.append({
                    "wall_candidate_id": str(wall["id"]),
                    "length_m": float(wall["length_m"]),
                    "boundary_distance_m": round(distance, 3),
                    "supporting_height_bands": wall.get("supporting_height_bands", []),
                    "status": "unverified_spatial_association",
                })
        nearby_gaps = []
        for gap in wall_gap_candidates:
            distance = _distance_to_polygon(
                np.asarray(gap["center_xz_m"], dtype=np.float64), polygon
            )
            if distance <= maximum_boundary_distance_m:
                nearby_gaps.append({
                    "wall_gap_candidate_id": str(gap["id"]),
                    "observed_width_m": float(gap["observed_gap_m"]),
                    "boundary_distance_m": round(distance, 3),
                    "status": "unverified_spatial_association",
                })
        nearest_gap = min(
            nearby_gaps,
            key=lambda item: item["boundary_distance_m"],
            default=None,
        )
        if nearest_gap is None and wall_gap_candidates:
            nearest = min(
                wall_gap_candidates,
                key=lambda gap: _distance_to_polygon(
                    np.asarray(gap["center_xz_m"], dtype=np.float64), polygon
                ),
            )
            nearest_gap = {
                "wall_gap_candidate_id": str(nearest["id"]),
                "observed_width_m": float(nearest["observed_gap_m"]),
                "boundary_distance_m": round(
                    _distance_to_polygon(
                        np.asarray(nearest["center_xz_m"], dtype=np.float64), polygon
                    ),
                    3,
                ),
                "status": "unverified_spatial_association",
            }
        nearby_walls.sort(key=lambda item: item["boundary_distance_m"])
        nearby_gaps.sort(key=lambda item: item["boundary_distance_m"])
        enriched.append({
            **room,
            "nearby_wall_evidence": nearby_walls[:8],
            "nearby_wall_gap_evidence": nearby_gaps[:4],
            "nearest_wall_gap_evidence": nearest_gap,
        })
    return enriched
