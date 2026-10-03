"""Versioned result document for the room reconstruction command."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.7.0"


def build_plan_document(
    *, capture: str, point_count: int, frame_step: int, voxel_size_m: float,
    wall_candidates: list[dict[str, object]] | None = None,
    cross_height_wall_candidates: list[dict[str, object]] | None = None,
    room_candidates: list[dict[str, object]] | None = None,
    opening_candidates: list[dict[str, object]] | None = None,
    wall_gap_candidates: list[dict[str, object]] | None = None,
    wall_room_candidates: list[dict[str, object]] | None = None,
    partial_wall_room_candidates: list[dict[str, object]] | None = None,
    pass_split: dict[str, object] | None = None,
    surface_candidates: dict[str, object] | None = None,
    ground_truth_comparison: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Describe what the current LiDAR milestone measured and what it did not."""
    surface_candidates = surface_candidates or {
        "floor_height_candidate": None,
        "ceiling_height_candidates": [],
        "ceiling_candidate_status": "insufficient_data",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture": {"id": capture, "tier": "lidar"},
        "processing": {
            "frame_step": frame_step,
            "voxel_size_m": voxel_size_m,
            "point_count": point_count,
            "pass_split": pass_split,
        },
        "rooms": build_room_drafts(room_candidates or []),
        "wall_candidates": wall_candidates or [],
        "cross_height_wall_candidates": cross_height_wall_candidates or [],
        "room_candidates": room_candidates or [],
        "opening_candidates": opening_candidates or [],
        "wall_gap_candidates": wall_gap_candidates or [],
        "wall_room_candidates": wall_room_candidates or [],
        "partial_wall_room_candidates": partial_wall_room_candidates or [],
        "floor_height_candidate": surface_candidates["floor_height_candidate"],
        "ceiling_height_candidates": surface_candidates["ceiling_height_candidates"],
        "ground_truth_comparison": ground_truth_comparison,
        "room_connections": [
            {"room_ids": candidate["room_ids"], "opening_candidate_id": candidate["id"]}
            for candidate in opening_candidates or []
        ],
        "scope_line_items": [],
        "capabilities": {
            "point_cloud": "available",
            "room_segmentation": "candidate_available",
            "walls": "candidate_available",
            "cross_height_wall_support": "candidate_available",
            "wall_gaps": "candidate_available",
            "wall_supported_room_outlines": "candidate_available",
            "partial_wall_outlines": "candidate_available",
            "floor_area": "not_implemented",
            "room_surface_measurements": "candidate_available",
            "floor_elevation": (
                "candidate_available" if surface_candidates["floor_height_candidate"]
                else "insufficient_data"
            ),
            "ceiling": surface_candidates["ceiling_candidate_status"],
            "openings": "candidate_available" if wall_gap_candidates else "insufficient_data",
            "room_adjacency": "not_implemented",
            "damage": "not_implemented",
            "concealed_damage": "not_implemented",
            "scope_estimation": "not_implemented",
            "photos_tier": "candidate_available",
            "video_tier": "candidate_available",
            "visual_overlap": "candidate_available",
        },
        "artifacts": {
            "point_cloud": "point_cloud.npz",
            "preview": "plan.png",
            "wall_candidates_preview": "wall_candidates.png",
            "cross_height_wall_candidates": "cross_height_wall_candidates.json",
            "cross_height_wall_candidates_preview": "cross_height_wall_candidates.png",
            "room_candidates_preview": "room_candidates.png",
            "room_layout_draft": "room_layout_draft.png",
            "opening_candidates": "opening_candidates.json",
            "wall_gap_candidates": "wall_gap_candidates.json",
            "wall_room_candidates": "wall_room_candidates.json",
            "partial_wall_room_candidates": "partial_wall_room_candidates.json",
            "vertical_surface_candidates": "vertical_surface_candidates.json",
            **(
                {"ground_truth_comparison": "ground_truth_comparison.json"}
                if ground_truth_comparison is not None else {}
            ),
        },
        "limitations": [
            "The preview shows a projected point cloud; it is not a floor plan.",
            "Wall traces are geometric candidates and require visual confirmation.",
            "Wall support counts show the strongest overlapping subsection across height bands, not an accuracy score.",
            "Wall gaps are geometric interruptions and are not confirmed doorways.",
            "Wall-supported room outlines require visual and measured review; partial side coverage does not prove a complete wall.",
            "Floor elevation is a pass-level candidate; ceiling-height peaks are not assigned to rooms or confirmed as ceilings.",
            "Photo/video tiers provide image-space overlap diagnostics only; they do not produce metric 3D geometry.",
            "Room records are unverified raster candidates; their intervals show processing sensitivity, not calibrated confidence or accuracy.",
            "Openings, adjacency, damage, and scope items are not confirmed by this capture.",
        ],
    }


def build_room_drafts(
    room_candidates: list[dict[str, object]],
    *,
    grid_size_m: float = 0.08,
    gap_closing_cells: int = 5,
) -> list[dict[str, object]]:
    """Expose candidate geometry as useful room records with explicit limits.

    The area interval is a boundary sensitivity band based on the raster's
    gap-closing radius. It is not a statistical confidence interval and does
    not include systematic pose drift. Wall trace lengths use the same broad
    sensitivity band and remain unverified associations.
    """
    boundary_sensitivity_m = grid_size_m * gap_closing_cells
    rooms = []
    for candidate in room_candidates:
        geometry = candidate.get("geometry_candidate") or {}
        area = float(geometry.get("polygon_area_m2", candidate.get("estimated_area_m2", 0.0)))
        perimeter = float(geometry.get("polygon_perimeter_m", 0.0))
        if area <= 0:
            continue
        delta_area = perimeter * boundary_sensitivity_m
        area_interval = [
            round(max(0.0, area - delta_area), 2),
            round(area + delta_area + 3.14159 * boundary_sensitivity_m**2, 2),
        ]
        floor_area = {
            "value": round(area, 2),
            "unit": "m²",
            "interval": area_interval,
            "status": "estimated",
            "reason": (
                f"Raster area candidate. Interval is a ±{boundary_sensitivity_m:.2f} m "
                "boundary sensitivity band from gap closing, not calibrated confidence; "
                "pose drift and missing walls are not covered."
            ),
        }
        walls = []
        for evidence in candidate.get("nearby_wall_evidence", []):
            length = float(evidence["length_m"])
            walls.append({
                "id": str(evidence["wall_candidate_id"]),
                "length_m": {
                    "value": round(length, 3),
                    "unit": "m",
                    "interval": [
                        round(max(0.0, length - boundary_sensitivity_m), 3),
                        round(length + boundary_sensitivity_m, 3),
                    ],
                    "status": "estimated",
                    "reason": (
                        "Nearby wall-trace length with an unverified association to this region; "
                        "interval is a processing sensitivity band, not calibrated confidence."
                    ),
                },
                "status": "candidate_unverified",
            })
        rooms.append({
            "id": str(candidate["id"]),
            "status": "candidate_unverified",
            "floor_polygon_m": candidate.get("boundary_xz_m", []),
            "floor_area_m2": floor_area,
            "ceiling_height_m": {
                "value": None,
                "unit": "m",
                "interval": None,
                "status": "insufficient_data",
                "reason": (
                    "Upper height peaks exist but are not confirmed as a ceiling for this room; "
                    "no defensible per-room ceiling measurement is available."
                ),
            },
            "walls": walls,
            "openings": [],
            "adjacent_room_ids": [],
            "damage_regions": [],
            "concealed_damage": [{
                "type": "unknown",
                "status": "not_assessed",
                "reason": "Damage is outside the current point-cloud analysis.",
            }],
        })
    return rooms


def capture_label(path: Path) -> str:
    """Return a stable, readable capture label from a ZIP or capture directory."""
    if path.suffix.lower() == ".zip":
        return path.stem
    return path.parent.name or path.name
