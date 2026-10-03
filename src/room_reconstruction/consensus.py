"""Geometric agreement between wall candidates at different heights."""

from __future__ import annotations

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
