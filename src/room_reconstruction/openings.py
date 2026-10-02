"""Possible openings inferred from gaps between aligned wall candidates."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def infer_opening_candidates(
    wall_candidates: list[dict[str, object]],
    room_candidates: list[dict[str, object]],
    *,
    min_gap_m: float = 0.4,
    max_gap_m: float = 1.3,
    max_offset_m: float = 0.2,
    max_center_distance_m: float = 4.5,
) -> list[dict[str, object]]:
    """Find wall gaps that lie between two room candidates on opposite sides.

    This is a conservative adjacency hypothesis. It does not identify a gap as
    a door or report a measured opening width.
    """
    if min_gap_m <= 0 or max_gap_m < min_gap_m or max_offset_m <= 0:
        raise ValueError("opening gap and offset limits must be positive and ordered")
    if max_center_distance_m <= 0:
        raise ValueError("max_center_distance_m must be positive")

    found: list[dict[str, object]] = []
    for first, second in combinations(wall_candidates, 2):
        p = np.asarray(first["start_xz_m"], dtype=np.float64)
        q = np.asarray(first["end_xz_m"], dtype=np.float64)
        r = np.asarray(second["start_xz_m"], dtype=np.float64)
        s = np.asarray(second["end_xz_m"], dtype=np.float64)
        first_vector = q - p
        length = np.linalg.norm(first_vector)
        if length <= 0:
            continue
        direction = first_vector / length
        second_vector = s - r
        second_length = np.linalg.norm(second_vector)
        if second_length <= 0 or abs(float(np.dot(direction, second_vector / second_length))) < 0.995:
            continue
        normal = np.array([-direction[1], direction[0]])
        first_offset = float(np.dot((p + q) / 2, normal))
        second_offset = float(np.dot((r + s) / 2, normal))
        if abs(first_offset - second_offset) > max_offset_m:
            continue

        first_interval = sorted((float(np.dot(p, direction)), float(np.dot(q, direction))))
        second_interval = sorted((float(np.dot(r, direction)), float(np.dot(s, direction))))
        if first_interval[1] < second_interval[0]:
            gap = second_interval[0] - first_interval[1]
            gap_start_t, gap_end_t = first_interval[1], second_interval[0]
        elif second_interval[1] < first_interval[0]:
            gap = first_interval[0] - second_interval[1]
            gap_start_t, gap_end_t = second_interval[1], first_interval[0]
        else:
            continue
        if not min_gap_m <= gap <= max_gap_m:
            continue
        center_t = (gap_start_t + gap_end_t) / 2
        midpoint = direction * center_t + normal * ((first_offset + second_offset) / 2)
        gap_endpoints = [
            direction * gap_start_t + normal * ((first_offset + second_offset) / 2),
            direction * gap_end_t + normal * ((first_offset + second_offset) / 2),
        ]

        room_sides = []
        for room in room_candidates:
            center = np.asarray(room["center_xz_m"], dtype=np.float64)
            signed_side = float(np.dot(center - midpoint, normal))
            distance = float(np.linalg.norm(center - midpoint))
            if distance <= max_center_distance_m and abs(signed_side) > 0.25:
                room_sides.append((room, signed_side, distance))
        positive = [item for item in room_sides if item[1] > 0]
        negative = [item for item in room_sides if item[1] < 0]
        if not positive or not negative:
            continue
        first_room = min(positive, key=lambda item: item[2])[0]
        second_room = min(negative, key=lambda item: item[2])[0]
        room_ids = sorted((str(first_room["id"]), str(second_room["id"])) )
        if any(
            item["room_ids"] == room_ids
            and np.linalg.norm(np.asarray(item["center_xz_m"]) - midpoint) < 0.3
            for item in found
        ):
            continue
        found.append({
            "id": f"opening_candidate_{len(found) + 1:02d}",
            "room_ids": room_ids,
            "center_xz_m": [round(float(value), 3) for value in midpoint],
            "gap_endpoints_xz_m": [
                [round(float(value), 3) for value in endpoint]
                for endpoint in gap_endpoints
            ],
            "observed_gap_m": round(float(gap), 3),
            "status": "candidate_unverified",
            "reason": (
                "Gap between aligned wall traces with room candidates on both sides; "
                "may be a doorway or missing scan data."
            ),
        })
    return found
