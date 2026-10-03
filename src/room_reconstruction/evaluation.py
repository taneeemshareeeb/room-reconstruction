"""Compare unregistered scan candidates with manually recorded measurements."""

from __future__ import annotations

import itertools

def compare_to_ground_truth(
    room_candidates: list[dict[str, object]],
    wall_gap_candidates: list[dict[str, object]],
    ground_truth: dict[str, object],
) -> dict[str, object]:
    """Report nearest numerical matches while preserving the lack of registration."""
    room = ground_truth.get("room", {})
    if not isinstance(room, dict):
        raise ValueError("ground-truth document must contain a room object")
    if "top_width" in room:
        widths = [float(room["top_width"])]
    else:
        widths = [float(room[key]) for key in ("width_top", "width_bottom") if key in room]
    spans_by_room: list[tuple[str, list[float]]] = []
    for candidate in room_candidates:
        geometry = candidate.get("geometry_candidate")
        if not isinstance(geometry, dict):
            continue
        spans = [float(value) for value in geometry.get("dominant_axis_spans_m", [])]
        if len(spans) == 2:
            spans_by_room.append((str(candidate["id"]), spans))

    dimension_matches = []
    width_pair_ranking = []
    for room_id, spans in spans_by_room:
        for name in ("left_length", "right_length"):
            measurement = room.get(name)
            if not isinstance(measurement, dict) or "value" not in measurement:
                continue
            measured_value = float(measurement["value"])
            closest_span = min(spans, key=lambda value: abs(value - measured_value))
            dimension_matches.append({
                "measurement": name,
                "measured_m": measured_value,
                "closest_room_candidate_id": room_id,
                "closest_rough_span_m": round(closest_span, 2),
                "absolute_difference_m": round(abs(closest_span - measured_value), 2),
                "status": "unverified_dimension_comparison",
            })
        if widths:
            errors = [
                sum(abs(float(span) - float(target)) for span, target in zip(spans, assignment))
                / len(assignment)
                for assignment in itertools.permutations(widths)
            ]
            if len(widths) == 1:
                errors = [min(abs(span - widths[0]) for span in spans)]
            width_pair_ranking.append({
                "room_candidate_id": room_id,
                "rough_spans_m": [round(value, 2) for value in sorted(spans, reverse=True)],
                "mean_absolute_width_difference_m": round(min(errors), 3),
            })
    width_pair_ranking.sort(key=lambda item: item["mean_absolute_width_difference_m"])
    closest_side_matches = []
    for measurement_name in ("left_length", "right_length"):
        matches = [item for item in dimension_matches if item["measurement"] == measurement_name]
        if matches:
            closest_side_matches.append(min(matches, key=lambda item: item["absolute_difference_m"]))

    doorway = room.get("doorway", {})
    doorway_width = float(doorway["width"]) if isinstance(doorway, dict) and "width" in doorway else None
    closest_gap = min(
        wall_gap_candidates,
        key=lambda item: abs(float(item["observed_gap_m"]) - doorway_width),
        default=None,
    ) if doorway_width is not None else None
    doorway_comparison = None
    if doorway_width is not None:
        doorway_comparison = {
            "measured_width_m": doorway_width,
            "closest_gap_candidate_id": str(closest_gap["id"]) if closest_gap else None,
            "candidate_width_m": float(closest_gap["observed_gap_m"]) if closest_gap else None,
            "absolute_difference_m": round(abs(float(closest_gap["observed_gap_m"]) - doorway_width), 3) if closest_gap else None,
            "relative_difference_fraction": round(abs(float(closest_gap["observed_gap_m"]) - doorway_width) / doorway_width, 4) if closest_gap else None,
            "status": "unverified_geometric_match" if closest_gap else "insufficient_data",
            "reason": "Width similarity alone does not establish that the gap is the measured doorway; the scan and sketch are not spatially registered.",
        }

    partial_bottom_wall = room.get("bottom_wall_segment_before_doorway", {})
    bottom_wall_record = None
    if isinstance(partial_bottom_wall, dict) and "value" in partial_bottom_wall:
        bottom_wall_record = {
            "measured_m": float(partial_bottom_wall["value"]),
            "status": "recorded_partial_segment_not_spatially_registered",
            "reason": "This is a partial wall before the doorway; no scan segment has been registered to it.",
        }

    ceiling_height = float(room["ceiling_height"]) if "ceiling_height" in room else None
    ceiling_matches = []
    if ceiling_height is not None:
        for candidate in room_candidates:
            surfaces = candidate.get("upper_surface_candidates", {})
            if not isinstance(surfaces, dict):
                continue
            for peak in surfaces.get("height_candidates", []):
                value = float(peak["height_above_local_floor_m"])
                ceiling_matches.append({
                    "room_candidate_id": str(candidate["id"]),
                    "candidate_height_m": value,
                    "absolute_difference_m": round(abs(value - ceiling_height), 3),
                    "supporting_points": int(peak["supporting_points"]),
                    "status": "unverified_height_match",
                })
    ceiling_matches.sort(key=lambda item: item["absolute_difference_m"])

    return {
        "status": "partial_unregistered_comparison",
        "ground_truth_source": ground_truth.get("source", "user-provided measurements"),
        "spatial_registration": "not_available",
        "measured_side_comparisons": dimension_matches,
        "closest_candidate_for_each_measured_side": closest_side_matches,
        "room_candidate_ranking_by_widths": width_pair_ranking,
        "doorway_comparison": doorway_comparison,
        "bottom_wall_segment": bottom_wall_record,
        "ceiling_height_comparison": {
            "measured_m": ceiling_height,
            "closest_room_candidate_peak": ceiling_matches[0] if ceiling_matches else None,
            "all_room_peak_comparisons": ceiling_matches,
            "status": "unverified_height_match" if ceiling_matches else "insufficient_data",
        },
        "limitations": [
            "Rough room spans are not fitted wall lengths and cannot establish which scan region is the measured room.",
            "The measurement sketch and scan have no shared coordinate registration.",
            "A close doorway width or ceiling height is numerical similarity only, not confirmation of identity.",
        ],
    }
