#!/usr/bin/env python3
"""Summarize ZIP archives without extracting their contents."""

from __future__ import annotations

import argparse
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_ARCHIVES = (
    Path("data/raw/single_room.zip"),
    Path("data/raw/single_scan_floor_only.zip"),
    Path("data/raw/single_scan_with_ceiling.zip"),
)
DEFAULT_SAMPLE_SIZE = 5


def is_suspicious_path(name: str) -> bool:
    """Return whether a ZIP member name is absolute or traverses upward."""
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    has_drive_prefix = bool(re.match(r"^[a-zA-Z]:", normalized))
    return path.is_absolute() or has_drive_prefix or ".." in path.parts


def inspect_archive(path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE) -> dict:
    """Read ZIP metadata and return a summary; never read or extract file data."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()

    extensions = Counter()
    samples = []
    suspicious = []
    total_size = 0
    for entry in entries:
        if entry.is_dir():
            extension = "[directory]"
        else:
            suffix = Path(entry.filename.replace("\\", "/")).suffix.lower()
            extension = suffix if suffix else "[no extension]"
        extensions[extension] += 1
        total_size += entry.file_size
        if len(samples) < sample_size:
            samples.append(entry.filename)
        if is_suspicious_path(entry.filename):
            suspicious.append(entry.filename)

    return {
        "path": path,
        "entry_count": len(entries),
        "total_uncompressed_size": total_size,
        "extensions": extensions,
        "sample_paths": samples,
        "suspicious_paths": suspicious,
    }


def format_size(size: int) -> str:
    return f"{size:,} bytes"


def print_summary(path: Path, sample_size: int) -> bool:
    try:
        summary = inspect_archive(path, sample_size)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as error:
        print(f"Archive: {path}\n  ERROR: cannot inspect archive: {error}\n")
        return False

    print(f"Archive: {path}")
    print(f"  Entries: {summary['entry_count']:,}")
    print(f"  Total uncompressed size: {format_size(summary['total_uncompressed_size'])}")
    print("  Extensions:")
    if summary["extensions"]:
        for extension, count in sorted(summary["extensions"].items()):
            print(f"    {extension}: {count:,}")
    else:
        print("    (none)")
    print(f"  Sample paths (up to {sample_size}):")
    if summary["sample_paths"]:
        for member in summary["sample_paths"]:
            print(f"    {member}")
    else:
        print("    (none)")
    print(f"  Suspicious paths: {len(summary['suspicious_paths']):,}")
    for member in summary["suspicious_paths"]:
        print(f"    {member}")
    print()
    return True


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect ZIP metadata without extracting archive contents."
    )
    parser.add_argument(
        "archives",
        nargs="*",
        type=Path,
        default=DEFAULT_ARCHIVES,
        help="ZIP files to inspect (defaults to the three data/raw archives)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"number of initial member paths to show (default: {DEFAULT_SAMPLE_SIZE})",
    )
    args = parser.parse_args(argv)
    if args.sample_size < 0:
        parser.error("--sample-size must be zero or greater")

    succeeded = True
    for archive_path in args.archives:
        if not print_summary(archive_path, args.sample_size):
            succeeded = False
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
