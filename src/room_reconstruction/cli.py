"""Command-line runner for the first LiDAR reconstruction milestone."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from .consensus import (
    annotate_current_wall_support,
    detect_wall_candidates_by_height,
    find_consensus_walls,
    propose_wall_supported_rooms,
)
from .openings import infer_opening_candidates, infer_wall_gap_candidates
from .plan import build_plan_document, capture_label
from .pointcloud import load_archive_point_cloud, load_capture_point_cloud
from .render import (
    render_point_cloud_preview,
    render_room_candidate_preview,
    render_wall_consensus_preview,
    render_wall_candidate_preview,
)
from .rooms import detect_room_candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a point-cloud preview and versioned JSON result from a LiDAR capture."
    )
    parser.add_argument("capture", type=Path, help="capture directory or capture ZIP")
    parser.add_argument("--tier", choices=("photos", "video", "lidar"), default="lidar")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.02, help="voxel width in metres")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    args = parser.parse_args(argv)

    if args.tier != "lidar":
        parser.error(f"the {args.tier} tier is not implemented yet; currently supported: lidar")
    try:
        loader = load_archive_point_cloud if args.capture.suffix.lower() == ".zip" else load_capture_point_cloud
        points = loader(args.capture, frame_step=args.frame_step, voxel_size=args.voxel_size)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_dir / "point_cloud.npz",
            points=points,
            frame_step=np.int64(args.frame_step),
            voxel_size_m=np.float64(args.voxel_size),
        )
        render_point_cloud_preview(
            points, str(args.output_dir / "plan.png"), capture_label(args.capture)
        )
        candidates_by_band = detect_wall_candidates_by_height(points)
        cross_height_wall_candidates = find_consensus_walls(candidates_by_band)
        wall_room_candidates = propose_wall_supported_rooms(cross_height_wall_candidates)
        wall_candidates = annotate_current_wall_support(
            candidates_by_band["current"], cross_height_wall_candidates
        )
        (args.output_dir / "wall_candidates.json").write_text(
            json.dumps(wall_candidates, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "cross_height_wall_candidates.json").write_text(
            json.dumps(cross_height_wall_candidates, indent=2) + "\n", encoding="utf-8"
        )
        render_wall_candidate_preview(
            points, wall_candidates, str(args.output_dir / "wall_candidates.png"),
            capture_label(args.capture),
        )
        render_wall_consensus_preview(
            points,
            cross_height_wall_candidates,
            str(args.output_dir / "cross_height_wall_candidates.png"),
            capture_label(args.capture),
        )
        room_candidates = detect_room_candidates(points)
        opening_candidates = infer_opening_candidates(wall_candidates, room_candidates)
        wall_gap_candidates = infer_wall_gap_candidates(wall_candidates)
        (args.output_dir / "room_candidates.json").write_text(
            json.dumps(room_candidates, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "opening_candidates.json").write_text(
            json.dumps(opening_candidates, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "wall_gap_candidates.json").write_text(
            json.dumps(wall_gap_candidates, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "wall_room_candidates.json").write_text(
            json.dumps(wall_room_candidates, indent=2) + "\n", encoding="utf-8"
        )
        render_room_candidate_preview(
            points, room_candidates, str(args.output_dir / "room_candidates.png"),
            capture_label(args.capture), opening_candidates=opening_candidates,
            wall_gap_candidates=wall_gap_candidates,
            wall_room_candidates=wall_room_candidates,
        )
        document = build_plan_document(
            capture=capture_label(args.capture), point_count=len(points),
            frame_step=args.frame_step, voxel_size_m=args.voxel_size,
            wall_candidates=wall_candidates,
            cross_height_wall_candidates=cross_height_wall_candidates,
            room_candidates=room_candidates,
            opening_candidates=opening_candidates,
            wall_gap_candidates=wall_gap_candidates,
            wall_room_candidates=wall_room_candidates,
        )
        (args.output_dir / "plan.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"Could not process capture: {exc}", file=sys.stderr)
        return 1

    print(
        f"Status: partial ({len(wall_candidates)} wall, {len(room_candidates)} raster room, "
        f"{len(wall_room_candidates)} wall-supported outline, "
        f"{len(opening_candidates)} room-opening, {len(wall_gap_candidates)} wall-gap, "
        f"{len(cross_height_wall_candidates)} "
        "cross-height wall candidates)"
    )
    print(f"Result: {args.output_dir / 'plan.json'}")
    print(f"Preview: {args.output_dir / 'plan.png'}")
    print(f"Point cloud: {args.output_dir / 'point_cloud.npz'}")
    print(f"Wall candidate preview: {args.output_dir / 'wall_candidates.png'}")
    print(f"Cross-height wall support: {args.output_dir / 'cross_height_wall_candidates.json'}")
    print(f"Cross-height wall preview: {args.output_dir / 'cross_height_wall_candidates.png'}")
    print(f"Room candidate preview: {args.output_dir / 'room_candidates.png'}")
    print(f"Opening candidates: {args.output_dir / 'opening_candidates.json'}")
    print(f"Wall-gap candidates: {args.output_dir / 'wall_gap_candidates.json'}")
    print(f"Wall-supported room outlines: {args.output_dir / 'wall_room_candidates.json'}")
    return 0
