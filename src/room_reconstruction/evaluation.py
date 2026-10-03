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
    closest_nearby_wall_for_each_side = []
    for measurement_name in ("left_length", "right_length"):
        matches = [item for item in dimension_matches if item["measurement"] == measurement_name]
        if matches:
            closest_side_matches.append(min(matches, key=lambda item: item["absolute_difference_m"]))
        measured_value = room.get(measurement_name)
        if not isinstance(measured_value, dict) or "value" not in measured_value:
            continue
        wall_matches = []
        for candidate in room_candidates:
            for wall in candidate.get("nearby_wall_evidence", []):
                wall_matches.append({
                    "measurement": measurement_name,
                    "measured_m": float(measured_value["value"]),
                    "room_candidate_id": str(candidate["id"]),
                    "wall_candidate_id": str(wall["wall_candidate_id"]),
                    "candidate_wall_length_m": float(wall["length_m"]),
                    "absolute_difference_m": round(
                        abs(float(wall["length_m"]) - float(measured_value["value"])), 3
                    ),
                    "boundary_distance_m": float(wall["boundary_distance_m"]),
                    "supporting_height_bands": wall["supporting_height_bands"],
                    "status": "unverified_local_wall_length_match",
                })
        if wall_matches:
            closest_nearby_wall_for_each_side.append(
                min(wall_matches, key=lambda item: item["absolute_difference_m"])
            )

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
    room_gap_proximity = []
    room_distinct_wall_checks = []
    for candidate in room_candidates:
        nearest = candidate.get("nearest_wall_gap_evidence")
        if nearest is not None:
            room_gap_proximity.append({
                "room_candidate_id": str(candidate["id"]),
                "wall_gap_candidate_id": str(nearest["wall_gap_candidate_id"]),
                "observed_width_m": float(nearest["observed_width_m"]),
                "distance_to_region_boundary_m": float(nearest["boundary_distance_m"]),
                "status": "unverified_spatial_association",
            })
        side_measurements = []
        for side_name in ("left_length", "right_length"):
            side = room.get(side_name)
            if isinstance(side, dict) and "value" in side:
                side_measurements.append((side_name, float(side["value"])))
        nearby_walls = candidate.get("nearby_wall_evidence", [])
        assignment_size = min(len(side_measurements), len(nearby_walls))
        assignments = []
        if assignment_size:
            for selected in itertools.permutations(nearby_walls, assignment_size):
                for target_indices in itertools.combinations(range(len(side_measurements)), assignment_size):
                    pairs = [
                        (side_measurements[index], wall)
                        for index, wall in zip(target_indices, selected)
                    ]
                    error = sum(
                        abs(float(wall["length_m"]) - measured)
                        for (_, measured), wall in pairs
                    )
                    assignments.append((error, pairs))
        best_assignment = min(assignments, key=lambda item: item[0], default=(0.0, []))[1]
        distinct_matches = []
        matched_names = set()
        for (side_name, measured), wall in best_assignment:
            matched_names.add(side_name)
            difference = abs(float(wall["length_m"]) - measured)
            distinct_matches.append({
                "measurement": side_name,
                "measured_m": measured,
                "wall_candidate_id": str(wall["wall_candidate_id"]),
                "candidate_wall_length_m": float(wall["length_m"]),
                "absolute_difference_m": round(difference, 3),
                "boundary_distance_m": float(wall["boundary_distance_m"]),
                "supporting_height_bands": wall["supporting_height_bands"],
                "status": "length_compatible_candidate" if difference <= 0.2 else "weak_length_match",
            })
        for side_name, measured in side_measurements:
            if side_name not in matched_names:
                distinct_matches.append({
                    "measurement": side_name,
                    "measured_m": measured,
                    "wall_candidate_id": None,
                    "candidate_wall_length_m": None,
                    "absolute_difference_m": None,
                    "boundary_distance_m": None,
                    "supporting_height_bands": [],
                    "status": "insufficient_distinct_wall_evidence",
                })
        room_distinct_wall_checks.append({
            "room_candidate_id": str(candidate["id"]),
            "distinct_side_wall_match_count": sum(
                item["status"] == "length_compatible_candidate" for item in distinct_matches
            ),
            "side_measurement_count": len(side_measurements),
            "matches": distinct_matches,
            "status": "unverified_distinct_wall_assignment",
        })

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

    limitations = [
        "Rough room spans are not fitted wall lengths and cannot establish which scan region is the measured room.",
        "The measurement sketch and scan have no shared coordinate registration.",
        "A close doorway width or ceiling height is numerical similarity only, not confirmation of identity.",
    ]
    if (
        len(closest_nearby_wall_for_each_side) >= 2
        and closest_nearby_wall_for_each_side[0]["wall_candidate_id"]
        == closest_nearby_wall_for_each_side[1]["wall_candidate_id"]
    ):
        limitations.append(
            "The closest scan wall trace matches both side lengths numerically; it is one segment and cannot independently confirm two room sides."
        )
    return {
        "status": "partial_unregistered_comparison",
        "ground_truth_source": ground_truth.get("source", "user-provided measurements"),
        "spatial_registration": "not_available",
        "measured_side_comparisons": dimension_matches,
        "closest_candidate_for_each_measured_side": closest_side_matches,
        "closest_nearby_wall_for_each_side": closest_nearby_wall_for_each_side,
        "room_candidate_ranking_by_widths": width_pair_ranking,
        "doorway_comparison": doorway_comparison,
        "nearest_wall_gap_by_room_candidate": room_gap_proximity,
        "distinct_side_wall_checks_by_room_candidate": room_distinct_wall_checks,
        "bottom_wall_segment": bottom_wall_record,
        "ceiling_height_comparison": {
            "measured_m": ceiling_height,
            "closest_room_candidate_peak": ceiling_matches[0] if ceiling_matches else None,
            "all_room_peak_comparisons": ceiling_matches,
            "status": "unverified_height_match" if ceiling_matches else "insufficient_data",
        },
        "limitations": limitations,
    }
