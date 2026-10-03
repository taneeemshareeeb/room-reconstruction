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
    estimate_dominant_wall_axis_deg,
    find_consensus_walls,
    propose_wall_supported_rooms,
)
from .evaluation import compare_to_ground_truth
from .image_tiers import process_image_tier, visual_plan_document
from .openings import infer_opening_candidates, infer_wall_gap_candidates
from .passes import NoPassSplitError, detect_camera_pass_split
from .plan import build_plan_document, capture_label
from .pointcloud import load_archive_point_cloud, load_capture_point_cloud, voxel_downsample
from .render import (
    render_point_cloud_preview,
    render_room_candidate_preview,
    render_wall_consensus_preview,
    render_wall_candidate_preview,
)
from .rooms import detect_room_candidates
from .room_metrics import attach_nearby_wall_evidence, measure_room_regions
from .surfaces import detect_vertical_surface_candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a point-cloud preview and versioned JSON result from a LiDAR capture."
    )
    parser.add_argument("capture", type=Path, help="capture directory or capture ZIP")
    parser.add_argument("--tier", choices=("photos", "video", "lidar"), default="lidar")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--voxel-size", type=float, default=0.02, help="voxel width in metres")
    parser.add_argument("--sample-seconds", type=float, default=1.0,
                        help="video frame sampling interval for --tier video")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument(
        "--ground-truth", type=Path,
        help="optional JSON measurements to compare with scan candidates",
    )
    args = parser.parse_args(argv)

    if args.tier in {"photos", "video"}:
        try:
            analysis = process_image_tier(
                args.capture, tier=args.tier, output_dir=args.output_dir,
                sample_seconds=args.sample_seconds,
            )
            plan = visual_plan_document(
                tier=args.tier, capture=capture_label(args.capture), image_analysis=analysis
            )
            (args.output_dir / "plan.json").write_text(
                json.dumps(plan, indent=2) + "\n", encoding="utf-8"
            )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            print(f"Could not process {args.tier} capture: {exc}", file=sys.stderr)
            return 1
        print(
            f"Status: partial ({analysis['image_count']} images, "
            f"{analysis['registered_adjacent_pairs']} adjacent feature registrations; "
            "metric room geometry unavailable)"
        )
        print(f"Result: {args.output_dir / 'plan.json'}")
        print(f"Visual mosaic: {args.output_dir / 'plan.png'}")
        print(f"Overlap analysis: {args.output_dir / 'image_analysis.json'}")
        return 0
    try:
        loader = (
            load_archive_point_cloud
            if args.capture.suffix.lower() == ".zip"
            else load_capture_point_cloud
        )
        pass_split = None
        floor_pass_points = None
        ceiling_pass_points = None
        try:
            pass_split = detect_camera_pass_split(args.capture)
        except (NoPassSplitError, ValueError):
            pass
        if pass_split is not None:
            split_row = int(pass_split["split_row_index"])
            frame_count = int(pass_split["frame_count"])
            pass_b_start = split_row + (-split_row % args.frame_step)
            floor_pass_points = loader(
                args.capture,
                frame_step=args.frame_step,
                frame_range=(0, split_row),
                voxel_size=args.voxel_size,
            )
            ceiling_pass_points = loader(
                args.capture,
                frame_step=args.frame_step,
                frame_range=(pass_b_start, frame_count),
                voxel_size=args.voxel_size,
            )
            points = voxel_downsample(
                np.concatenate((floor_pass_points, ceiling_pass_points), axis=0),
                voxel_size=args.voxel_size,
            )
        else:
            floor_pass_points = loader(
                args.capture, frame_step=args.frame_step, voxel_size=args.voxel_size
            )
            points = floor_pass_points
        surface_candidates = detect_vertical_surface_candidates(
            floor_pass_points,
            ceiling_pass_points,
            floor_reference_pass="A" if pass_split is not None else "unsegmented",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "vertical_surface_candidates.json").write_text(
            json.dumps(surface_candidates, indent=2) + "\n", encoding="utf-8"
        )
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
        room_grid_angle = estimate_dominant_wall_axis_deg(cross_height_wall_candidates)
        room_candidates = detect_room_candidates(
            points, grid_rotation_deg=room_grid_angle
        )
        room_candidates = measure_room_regions(
            room_candidates,
            floor_pass_points,
            ceiling_pass_points,
            float(surface_candidates["floor_height_candidate"]["elevation_world_y_m"]),
            dominant_axis_deg=room_grid_angle,
        )
        wall_gap_candidates = infer_wall_gap_candidates(wall_candidates)
        room_candidates = attach_nearby_wall_evidence(
            room_candidates, wall_candidates, wall_gap_candidates
        )
        opening_candidates = infer_opening_candidates(wall_candidates, room_candidates)
        ground_truth_comparison = None
        if args.ground_truth is not None:
            ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
            ground_truth_comparison = compare_to_ground_truth(
                room_candidates, wall_gap_candidates, ground_truth
            )
            (args.output_dir / "ground_truth_comparison.json").write_text(
                json.dumps(ground_truth_comparison, indent=2) + "\n", encoding="utf-8"
            )
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
            pass_split=pass_split,
            surface_candidates=surface_candidates,
            ground_truth_comparison=ground_truth_comparison,
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
    print(f"Vertical surface candidates: {args.output_dir / 'vertical_surface_candidates.json'}")
    if args.ground_truth is not None:
        print(f"Ground-truth comparison: {args.output_dir / 'ground_truth_comparison.json'}")
    if pass_split is not None:
        print(
            f"Pass split: row {pass_split['split_row_index']} of "
            f"{pass_split['frame_count']} at {pass_split['elapsed_time_s']:.2f} s"
        )
    print(
        f"Height candidates: 1 floor, "
        f"{len(surface_candidates['ceiling_height_candidates'])} upper-surface"
    )
    return 0
