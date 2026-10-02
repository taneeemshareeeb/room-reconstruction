#!/usr/bin/env python3
"""Inspect pose continuity from capture ZIP metadata without extracting files."""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Iterable


DEFAULT_ARCHIVES = (
    Path("data/raw/single_room.zip"),
    Path("data/raw/single_scan_floor_only.zip"),
    Path("data/raw/single_scan_with_ceiling.zip"),
)
REQUIRED_COLUMNS = ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")


def _percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for non-empty values."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _parse_pose_rows(text: str, archive_path: Path) -> list[dict[str, object]]:
    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    if reader.fieldnames is None:
        raise ValueError("odometry.csv has no header")
    fields = {name.strip() for name in reader.fieldnames if name is not None}
    missing = sorted(set(REQUIRED_COLUMNS) - fields)
    if missing:
        raise ValueError(f"odometry.csv is missing columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            frame = int(row["frame"].strip())
            timestamp = float(row["timestamp"])
            position = tuple(float(row[axis]) for axis in ("x", "y", "z"))
            quaternion = tuple(float(row[key]) for key in ("qx", "qy", "qz", "qw"))
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid pose on CSV line {line_number}: {error}") from error
        values = (timestamp, *position, *quaternion)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite pose value on CSV line {line_number}")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise ValueError(f"zero-length quaternion on CSV line {line_number}")
        quaternion = tuple(value / norm for value in quaternion)
        rows.append({
            "frame": frame,
            "timestamp": timestamp,
            "position": position,
            "quaternion": quaternion,
        })

    if not rows:
        raise ValueError(f"{archive_path} contains no odometry rows")
    return rows


def inspect_trajectory(
    archive_path: Path,
    *,
    translation_jump_m: float = 0.25,
    rotation_jump_deg: float = 30.0,
) -> dict[str, object]:
    """Read odometry.csv from a ZIP and summarize pose continuity."""
    with zipfile.ZipFile(archive_path) as archive:
        odometry_paths = [
            name for name in archive.namelist()
            if Path(name).name.lower() == "odometry.csv"
        ]
        if len(odometry_paths) != 1:
            raise ValueError(
                f"expected exactly one odometry.csv, found {len(odometry_paths)}"
            )
        with archive.open(odometry_paths[0]) as stream:
            text = stream.read().decode("utf-8-sig")

    rows = _parse_pose_rows(text, archive_path)
    translations: list[float] = []
    rotations: list[float] = []
    jumps: list[tuple[int, int, float, float]] = []
    timestamp_order_errors = 0
    frame_order_errors = 0

    for previous, current in zip(rows, rows[1:]):
        p0 = previous["position"]
        p1 = current["position"]
        q0 = previous["quaternion"]
        q1 = current["quaternion"]
        distance = math.dist(p0, p1)
        dot = min(1.0, abs(sum(a * b for a, b in zip(q0, q1))))
        angle = math.degrees(2 * math.acos(dot))
        translations.append(distance)
        rotations.append(angle)
        if current["timestamp"] <= previous["timestamp"]:
            timestamp_order_errors += 1
        if current["frame"] <= previous["frame"]:
            frame_order_errors += 1
        if distance > translation_jump_m or angle > rotation_jump_deg:
            jumps.append((previous["frame"], current["frame"], distance, angle))

    positions = [row["position"] for row in rows]
    bounds = {
        axis: (min(position[index] for position in positions),
               max(position[index] for position in positions))
        for index, axis in enumerate(("x", "y", "z"))
    }
    return {
        "frame_count": len(rows),
        "duration_s": rows[-1]["timestamp"] - rows[0]["timestamp"],
        "bounds_m": bounds,
        "translation_m": {
            "median": statistics.median(translations) if translations else 0.0,
            "p95": _percentile(translations, 0.95) if translations else 0.0,
            "max": max(translations, default=0.0),
        },
        "rotation_deg": {
            "median": statistics.median(rotations) if rotations else 0.0,
            "p95": _percentile(rotations, 0.95) if rotations else 0.0,
            "max": max(rotations, default=0.0),
        },
        "jumps": jumps,
        "timestamp_order_errors": timestamp_order_errors,
        "frame_order_errors": frame_order_errors,
    }


def _print_report(
    archive_path: Path,
    *,
    translation_jump_m: float,
    rotation_jump_deg: float,
) -> bool:
    try:
        report = inspect_trajectory(
            archive_path,
            translation_jump_m=translation_jump_m,
            rotation_jump_deg=rotation_jump_deg,
        )
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        print(f"Capture: {archive_path}\n  ERROR: cannot inspect trajectory: {error}\n")
        return False

    print(f"Capture: {archive_path}")
    print(f"  Odometry frames: {report['frame_count']:,}")
    print(f"  Recorded duration: {report['duration_s']:.1f} s")
    print("  Position bounds (m):")
    for axis, (minimum, maximum) in report["bounds_m"].items():
        print(f"    {axis}: {minimum:.3f} to {maximum:.3f} (span {maximum - minimum:.3f})")
    translation = report["translation_m"]
    rotation = report["rotation_deg"]
    print(
        "  Frame-to-frame translation (m): "
        f"median {translation['median']:.3f}, p95 {translation['p95']:.3f}, "
        f"max {translation['max']:.3f}"
    )
    print(
        "  Frame-to-frame rotation (deg): "
        f"median {rotation['median']:.2f}, p95 {rotation['p95']:.2f}, "
        f"max {rotation['max']:.2f}"
    )
    print(
        f"  Large pose steps (> {translation_jump_m:g} m or "
        f"> {rotation_jump_deg:g} deg): {len(report['jumps'])}"
    )
    for first, second, distance, angle in report["jumps"][:10]:
        print(f"    frames {first}→{second}: {distance:.3f} m, {angle:.1f} deg")
    if len(report["jumps"]) > 10:
        print(f"    ... and {len(report['jumps']) - 10} more")
    print(
        "  Ordering errors: "
        f"{report['timestamp_order_errors']} timestamps, "
        f"{report['frame_order_errors']} frame IDs"
    )
    print("  Note: pose continuity is a capture diagnostic, not a drift or accuracy measurement.\n")
    return True


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize capture pose continuity without extracting ZIP contents."
    )
    parser.add_argument(
        "archives", nargs="*", type=Path, default=DEFAULT_ARCHIVES,
        help="capture ZIP files (defaults to the three data/raw archives)",
    )
    parser.add_argument("--translation-jump-m", type=float, default=0.25)
    parser.add_argument("--rotation-jump-deg", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.translation_jump_m <= 0 or args.rotation_jump_deg <= 0:
        parser.error("pose jump thresholds must be greater than zero")

    succeeded = True
    for archive_path in args.archives:
        if not _print_report(
            archive_path,
            translation_jump_m=args.translation_jump_m,
            rotation_jump_deg=args.rotation_jump_deg,
        ):
            succeeded = False
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
