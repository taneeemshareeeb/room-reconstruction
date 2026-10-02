"""Command-line runner for the first LiDAR reconstruction milestone."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from .plan import build_plan_document, capture_label
from .geometry import detect_wall_candidates
from .openings import infer_opening_candidates
from .pointcloud import load_archive_point_cloud, load_capture_point_cloud
from .render import (
    render_point_cloud_preview,
    render_room_candidate_preview,
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
        wall_candidates = detect_wall_candidates(points)
        (args.output_dir / "wall_candidates.json").write_text(
            json.dumps(wall_candidates, indent=2) + "\n", encoding="utf-8"
        )
        render_wall_candidate_preview(
            points, wall_candidates, str(args.output_dir / "wall_candidates.png"),
            capture_label(args.capture),
        )
        room_candidates = detect_room_candidates(points)
        opening_candidates = infer_opening_candidates(wall_candidates, room_candidates)
        (args.output_dir / "room_candidates.json").write_text(
            json.dumps(room_candidates, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "opening_candidates.json").write_text(
            json.dumps(opening_candidates, indent=2) + "\n", encoding="utf-8"
        )
        render_room_candidate_preview(
            points, room_candidates, str(args.output_dir / "room_candidates.png"),
            capture_label(args.capture), opening_candidates=opening_candidates,
        )
        document = build_plan_document(
            capture=capture_label(args.capture), point_count=len(points),
            frame_step=args.frame_step, voxel_size_m=args.voxel_size,
            wall_candidates=wall_candidates,
            room_candidates=room_candidates,
            opening_candidates=opening_candidates,
        )
        (args.output_dir / "plan.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"Could not process capture: {exc}", file=sys.stderr)
        return 1

    print(
        f"Status: partial ({len(wall_candidates)} wall, {len(room_candidates)} room, "
        f"{len(opening_candidates)} opening candidates)"
    )
    print(f"Result: {args.output_dir / 'plan.json'}")
    print(f"Preview: {args.output_dir / 'plan.png'}")
    print(f"Point cloud: {args.output_dir / 'point_cloud.npz'}")
    print(f"Wall candidate preview: {args.output_dir / 'wall_candidates.png'}")
    print(f"Room candidate preview: {args.output_dir / 'room_candidates.png'}")
    print(f"Opening candidates: {args.output_dir / 'opening_candidates.json'}")
    return 0
