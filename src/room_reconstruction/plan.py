"""Versioned result document for the room reconstruction command."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.4.0"


def build_plan_document(
    *, capture: str, point_count: int, frame_step: int, voxel_size_m: float,
    wall_candidates: list[dict[str, object]] | None = None,
    cross_height_wall_candidates: list[dict[str, object]] | None = None,
    room_candidates: list[dict[str, object]] | None = None,
    opening_candidates: list[dict[str, object]] | None = None,
    wall_gap_candidates: list[dict[str, object]] | None = None,
    wall_room_candidates: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Describe what the current LiDAR milestone measured and what it did not."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture": {"id": capture, "tier": "lidar"},
        "processing": {
            "frame_step": frame_step,
            "voxel_size_m": voxel_size_m,
            "point_count": point_count,
        },
        "rooms": [],
        "wall_candidates": wall_candidates or [],
        "cross_height_wall_candidates": cross_height_wall_candidates or [],
        "room_candidates": room_candidates or [],
        "opening_candidates": opening_candidates or [],
        "wall_gap_candidates": wall_gap_candidates or [],
        "wall_room_candidates": wall_room_candidates or [],
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
            "floor_area": "not_implemented",
            "ceiling": "not_implemented",
            "openings": "not_implemented",
            "room_adjacency": "not_implemented",
            "damage": "not_implemented",
            "concealed_damage": "not_implemented",
            "scope_estimation": "not_implemented",
            "photos_tier": "not_implemented",
            "video_tier": "not_implemented",
        },
        "artifacts": {
            "point_cloud": "point_cloud.npz",
            "preview": "plan.png",
            "wall_candidates_preview": "wall_candidates.png",
            "cross_height_wall_candidates": "cross_height_wall_candidates.json",
            "cross_height_wall_candidates_preview": "cross_height_wall_candidates.png",
            "room_candidates_preview": "room_candidates.png",
            "opening_candidates": "opening_candidates.json",
            "wall_gap_candidates": "wall_gap_candidates.json",
            "wall_room_candidates": "wall_room_candidates.json",
        },
        "limitations": [
            "The preview shows a projected point cloud; it is not a floor plan.",
            "Wall traces are geometric candidates and require visual confirmation.",
            "Wall support counts show the strongest overlapping subsection across height bands, not an accuracy score.",
            "Wall gaps are geometric interruptions and are not confirmed doorways.",
            "Wall-supported room outlines require visual and measured review; partial side coverage does not prove a complete wall.",
            "Verified room boundaries, dimensions, openings, damage, and scope items are not available.",
        ],
    }


def capture_label(path: Path) -> str:
    """Return a stable, readable capture label from a ZIP or capture directory."""
    if path.suffix.lower() == ".zip":
        return path.stem
    return path.parent.name or path.name
