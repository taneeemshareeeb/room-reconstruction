"""Geometric agreement between wall candidates at different heights."""

from __future__ import annotations

import itertools
import math

import numpy as np

from .geometry import detect_wall_candidates


WALL_HEIGHT_BANDS = (
    ("lower", 0.40, 0.80),
    ("current", 0.85, 1.55),
    ("upper", 1.50, 2.20),
)


def detect_wall_candidates_by_height(
    points: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    """Detect wall candidates independently in each standard height band."""
    return {
        label: detect_wall_candidates(
            points,
            slice_min_height_m=lower,
            slice_max_height_m=upper,
        )
        for label, lower, upper in WALL_HEIGHT_BANDS
    }


def _line_geometry(wall: dict[str, object]) -> dict[str, object]:
    start = np.asarray(wall["start_xz_m"], dtype=np.float64)
    end = np.asarray(wall["end_xz_m"], dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 0:
        raise ValueError("wall candidates must have positive length")
    direction = vector / length
    if direction[0] < 0 or (abs(direction[0]) < 1e-8 and direction[1] < 0):
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    offset = float(np.dot((start + end) / 2, normal))
    interval = sorted((float(np.dot(start, direction)), float(np.dot(end, direction))))
    return {
        "direction": direction,
        "normal": normal,
        "offset": offset,
        "interval": interval,
        "length": length,
    }


def _shared_overlap(
    group: dict[str, object],
    candidate: dict[str, object],
    *,
    max_angle_deg: float,
    max_offset_m: float,
    min_overlap_m: float,
) -> tuple[float, list[float]] | None:
    anchor = group["anchor"]
    direction = anchor["direction"]
    normal = anchor["normal"]
    if abs(float(np.dot(direction, candidate["direction"]))) < math.cos(math.radians(max_angle_deg)):
        return None

    wall = candidate["wall"]
    start = np.asarray(wall["start_xz_m"], dtype=np.float64)
    end = np.asarray(wall["end_xz_m"], dtype=np.float64)
    candidate_offset = float(np.dot((start + end) / 2, normal))
    if abs(candidate_offset - anchor["offset"]) > max_offset_m:
        return None
    interval = sorted((float(np.dot(start, direction)), float(np.dot(end, direction))))
    shared = [
        max(group["shared_interval"][0], interval[0]),
        min(group["shared_interval"][1], interval[1]),
    ]
    overlap = shared[1] - shared[0]
    required_overlap = max(min_overlap_m, 0.4 * min(anchor["length"], candidate["length"]))
    return (overlap, shared) if overlap >= required_overlap else None


def find_consensus_walls(
    candidates_by_band: dict[str, list[dict[str, object]]],
    *,
    max_angle_deg: float = 10.0,
    max_offset_m: float = 0.20,
    min_overlap_m: float = 0.75,
) -> list[dict[str, object]]:
    """Group overlapping, near-parallel wall candidates from distinct bands."""
    if (
        not math.isfinite(max_angle_deg)
        or not math.isfinite(max_offset_m)
        or not math.isfinite(min_overlap_m)
        or not 0 < max_angle_deg < 90
        or max_offset_m <= 0
        or min_overlap_m <= 0
    ):
        raise ValueError("consensus tolerances must be positive and angle below 90 degrees")

    groups: list[dict[str, object]] = []
    for band, *_ in WALL_HEIGHT_BANDS:
        for wall in sorted(
            candidates_by_band.get(band, []),
            key=lambda item: int(item.get("supporting_cells", 0)),
            reverse=True,
        ):
            geometry = _line_geometry(wall)
            item = {"band": band, "wall": wall, **geometry}
            matches = []
            for group in groups:
                if band in group["bands"]:
                    continue
                overlap = _shared_overlap(
                    group,
                    item,
                    max_angle_deg=max_angle_deg,
                    max_offset_m=max_offset_m,
                    min_overlap_m=min_overlap_m,
                )
                if overlap is not None:
                    matches.append((overlap[0], group, overlap[1]))
            if matches:
                _, group, shared_interval = max(matches, key=lambda result: result[0])
                group["members"].append(item)
                group["bands"].append(band)
                group["shared_interval"] = shared_interval
            else:
                groups.append({
                    "anchor": geometry,
                    "members": [item],
                    "bands": [band],
                    "shared_interval": list(geometry["interval"]),
                })

    consensus = []
    for group in groups:
        if len(group["bands"]) < 2:
            continue
        anchor = group["anchor"]
        start_t, end_t = group["shared_interval"]
        offset = float(np.mean([member["offset"] for member in group["members"]]))
        start = anchor["direction"] * start_t + anchor["normal"] * offset
        end = anchor["direction"] * end_t + anchor["normal"] * offset
        consensus.append({
            "id": f"wall_consensus_{len(consensus) + 1:02d}",
            "start_xz_m": [round(float(value), 3) for value in start],
            "end_xz_m": [round(float(value), 3) for value in end],
            "shared_length_m": round(float(end_t - start_t), 3),
            "supporting_bands": group["bands"],
            "support_count": len(group["bands"]),
            "source_candidates": [
                {"band": member["band"], "id": member["wall"]["id"]}
                for member in group["members"]
            ],
            "status": "candidate_unverified",
            "reason": (
                "Near-parallel wall candidates overlap in multiple height bands; "
                "agreement is not ground-truth validation."
            ),
        })
    return sorted(consensus, key=lambda item: (-item["support_count"], -item["shared_length_m"]))


def annotate_current_wall_support(
    current_walls: list[dict[str, object]],
    consensus_walls: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Add each current wall's strongest shared-subsegment evidence."""
    consensus_by_id = {str(wall["id"]): [] for wall in current_walls}
    for consensus in consensus_walls:
        for source in consensus["source_candidates"]:
            if source["band"] == "current" and source["id"] in consensus_by_id:
                consensus_by_id[source["id"]].append(consensus)

    annotated = []
    for wall in current_walls:
        matches = consensus_by_id[str(wall["id"])]
        strongest = max(
            matches,
            key=lambda item: (int(item["support_count"]), float(item["shared_length_m"])),
            default=None,
        )
        bands = strongest["supporting_bands"] if strongest else ["current"]
        annotated.append({
            **wall,
            "height_band_support_count": len(bands),
            "supporting_height_bands": bands,
            "cross_height_consensus_ids": [item["id"] for item in matches],
            "support_status": "unverified_consistency_hint",
        })
    return annotated


def estimate_dominant_wall_axis_deg(
    wall_candidates: list[dict[str, object]],
) -> float:
    """Estimate a grid orientation from wall directions without snapping them.

    Directions are averaged modulo 90 degrees so perpendicular wall families
    contribute to the same dominant raster orientation. Room contours remain
    free-form in world coordinates; this only rotates the analysis grid.
    """
    weighted_direction = 0j
    total_weight = 0.0
    for wall in wall_candidates:
        start = np.asarray(wall["start_xz_m"], dtype=np.float64)
        end = np.asarray(wall["end_xz_m"], dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 0:
            continue
        angle = math.atan2(float(vector[1]), float(vector[0]))
        weight = length * max(1, int(wall.get("support_count", 1)))
        weighted_direction += weight * np.exp(4j * angle)
        total_weight += weight
    if total_weight == 0 or abs(weighted_direction) < 1e-8:
        return 0.0
    angle_deg = math.degrees(math.atan2(weighted_direction.imag, weighted_direction.real) / 4)
    return float((angle_deg + 45.0) % 90.0 - 45.0)


def propose_wall_supported_rooms(
    consensus_walls: list[dict[str, object]],
    *,
    parallel_tolerance_deg: float = 12.0,
    perpendicular_tolerance_deg: float = 15.0,
    max_corner_deviation_deg: float = 15.0,
    min_side_m: float = 1.5,
    max_side_m: float = 7.0,
    min_pair_overlap_m: float = 0.5,
    min_side_coverage: float = 0.70,
) -> list[dict[str, object]]:
    """Propose four-sided outlines from supported, approximately rectangular walls.

    These are geometry hypotheses only. Every side must have measurable support
    from a cross-height consensus segment; unsupported portions are reported as
    coverage gaps rather than silently treated as observed wall.
    """
    if (
        not 0 < parallel_tolerance_deg < 45
        or not 0 < perpendicular_tolerance_deg < 45
        or not 0 < max_corner_deviation_deg < 45
        or min_side_m <= 0
        or max_side_m <= min_side_m
        or min_pair_overlap_m <= 0
        or not 0 < min_side_coverage <= 1
    ):
        raise ValueError("room-outline tolerances and dimensions are invalid")

    lines = []
    for wall in consensus_walls:
        start = np.asarray(wall["start_xz_m"], dtype=np.float64)
        end = np.asarray(wall["end_xz_m"], dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 0:
            continue
        direction = vector / length
        if direction[0] < 0 or (abs(direction[0]) < 1e-8 and direction[1] < 0):
            direction = -direction
        normal = np.array([-direction[1], direction[0]])
        lines.append({
            "wall": wall,
            "start": start,
            "end": end,
            "direction": direction,
            "normal": normal,
            "offset": float(np.dot((start + end) / 2, normal)),
            "interval": sorted((float(np.dot(start, direction)), float(np.dot(end, direction)))),
        })

    parallel_cosine = math.cos(math.radians(parallel_tolerance_deg))
    side_pairs = []
    for first, second in itertools.combinations(lines, 2):
        dot = float(np.dot(first["direction"], second["direction"]))
        if abs(dot) < parallel_cosine:
            continue
        normal = np.array([-first["direction"][1], first["direction"][0]])
        separation = abs(float(np.dot(second["start"] - first["start"], normal)))
        if not min_side_m <= separation <= max_side_m:
            continue
        first_interval = sorted((float(np.dot(first["start"], first["direction"])),
                                 float(np.dot(first["end"], first["direction"]))))
        second_interval = sorted((float(np.dot(second["start"], first["direction"])),
                                  float(np.dot(second["end"], first["direction"]))))
        overlap = min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0])
        if overlap < min_pair_overlap_m:
            continue
        side_pairs.append((first, second))

    def intersection(first: dict[str, object], second: dict[str, object]) -> np.ndarray | None:
        matrix = np.asarray([first["normal"], second["normal"]], dtype=np.float64)
        if abs(float(np.linalg.det(matrix))) < 1e-6:
            return None
        return np.linalg.solve(matrix, np.asarray([first["offset"], second["offset"]]))

    proposals = []
    seen_sources = set()
    for first_pair, second_pair in itertools.combinations(side_pairs, 2):
        first, opposite_a = first_pair
        perpendicular, opposite_b = second_pair
        source_ids = {
            str(first["wall"]["id"]), str(opposite_a["wall"]["id"]),
            str(perpendicular["wall"]["id"]), str(opposite_b["wall"]["id"]),
        }
        if len(source_ids) != 4 or tuple(sorted(source_ids)) in seen_sources:
            continue
        # The two opposing-wall families must meet at approximately right angles.
        axis_dot = abs(float(np.dot(first["direction"], perpendicular["direction"])))
        angle = math.degrees(math.acos(float(np.clip(axis_dot, -1.0, 1.0))))
        if abs(90.0 - angle) > perpendicular_tolerance_deg:
            continue

        corners = [
            intersection(first, perpendicular),
            intersection(opposite_a, perpendicular),
            intersection(opposite_a, opposite_b),
            intersection(first, opposite_b),
        ]
        if any(corner is None or not np.isfinite(corner).all() for corner in corners):
            continue
        corners_array = np.asarray(corners)
        side_lengths = [
            float(np.linalg.norm(corners_array[(index + 1) % 4] - corners_array[index]))
            for index in range(4)
        ]
        if any(length < min_side_m or length > max_side_m for length in side_lengths):
            continue
        corner_angles = []
        for index in range(4):
            incoming = corners_array[index - 1] - corners_array[index]
            outgoing = corners_array[(index + 1) % 4] - corners_array[index]
            cosine = float(np.dot(incoming, outgoing) / (np.linalg.norm(incoming) * np.linalg.norm(outgoing)))
            corner_angles.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
        if any(abs(angle_value - 90.0) > max_corner_deviation_deg
               for angle_value in corner_angles):
            continue

        side_lines = (perpendicular, opposite_a, opposite_b, first)
        coverage = []
        for index, line in enumerate(side_lines):
            side_start = corners_array[index]
            side_end = corners_array[(index + 1) % 4]
            direction = line["direction"]
            side_interval = sorted((float(np.dot(side_start, direction)), float(np.dot(side_end, direction))))
            support_interval = sorted((float(np.dot(line["start"], direction)),
                                       float(np.dot(line["end"], direction))))
            side_length = side_interval[1] - side_interval[0]
            covered = max(0.0, min(side_interval[1], support_interval[1])
                          - max(side_interval[0], support_interval[0]))
            coverage.append(covered / side_length if side_length > 0 else 0.0)
        if min(coverage) < min_side_coverage:
            continue

        area = abs(float(
            np.dot(corners_array[:, 0], np.roll(corners_array[:, 1], -1))
            - np.dot(corners_array[:, 1], np.roll(corners_array[:, 0], -1))
        )) / 2
        seen_sources.add(tuple(sorted(source_ids)))
        proposals.append({
            "id": "",
            "boundary_xz_m": [[round(float(value), 3) for value in corner] for corner in corners_array],
            "side_lengths_m": [round(length, 3) for length in side_lengths],
            "corner_angles_deg": [round(value, 1) for value in corner_angles],
            "estimated_area_m2": round(area, 2),
            "side_support_fraction": [round(value, 3) for value in coverage],
            "minimum_side_support_fraction": round(min(coverage), 3),
            "source_wall_ids": [
                str(line["wall"]["id"]) for line in side_lines
            ],
            "status": "candidate_unverified",
            "reason": (
                "Four cross-height wall segments form a closed four-sided outline. "
                "Side coverage is partial scan evidence, not ground-truth confirmation."
            ),
        })

    proposals.sort(key=lambda item: (
        -float(item["minimum_side_support_fraction"]),
        -sum(int(next(wall for wall in consensus_walls if wall["id"] == source)["support_count"])
             for source in item["source_wall_ids"]),
    ))
    for index, proposal in enumerate(proposals, start=1):
        proposal["id"] = f"wall_room_candidate_{index:02d}"
    return proposals
