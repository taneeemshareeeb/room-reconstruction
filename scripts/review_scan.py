#!/usr/bin/env python3
"""Serve a local, read-only review page for one reconstruction result folder."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = PROJECT_ROOT / "scripts" / "review_ui" / "index.html"
JSON_FILES = {
    "plan": "plan.json",
    "rooms": "room_candidates.json",
    "walls": "wall_candidates.json",
    "openings": "opening_candidates.json",
    "consensus": "cross_height_wall_candidates.json",
}
IMAGE_FILES = {
    "plan.png",
    "room_candidates.png",
    "wall_candidates.png",
    "cross_height_wall_candidates.png",
}


def load_results(results_dir: Path, report_prefix: str | None = None) -> dict[str, object]:
    """Read known JSON outputs, reporting missing or malformed artifacts clearly."""
    payload: dict[str, object] = {"files": {}, "errors": []}
    files = payload["files"]
    errors = payload["errors"]
    assert isinstance(files, dict) and isinstance(errors, list)
    if report_prefix is None:
        json_paths = {key: results_dir / filename for key, filename in JSON_FILES.items()}
        image_paths = {name: results_dir / name for name in IMAGE_FILES}
    else:
        json_paths = {
            "plan": None,
            "rooms": results_dir / f"{report_prefix}_room_candidates.json",
            "walls": results_dir / f"{report_prefix}_wall_candidates.json",
            "openings": results_dir / f"{report_prefix}_opening_candidates.json",
            "consensus": results_dir / f"{report_prefix}_wall_consensus.json",
        }
        image_paths = {
            "plan.png": results_dir / f"{report_prefix}_step10_voxel2cm.png",
            "room_candidates.png": results_dir / f"{report_prefix}_room_candidates.png",
            "wall_candidates.png": results_dir / f"{report_prefix}_wall_candidates.png",
            "cross_height_wall_candidates.png": results_dir / f"{report_prefix}_wall_consensus.png",
        }

    for key, path in json_paths.items():
        if path is None:
            files[key] = {
                "capture": {"id": report_prefix},
                "processing": {"frame_step": 10, "voxel_size_m": 0.02, "point_count": None},
            }
            continue
        if not path.is_file():
            files[key] = None
            errors.append(f"Missing {path.name}")
            continue
        try:
            files[key] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            files[key] = None
            errors.append(f"Could not read {path.name}: {exc}")

    # Earlier report JSON predates the per-wall support fields. Recover those
    # display hints from the matching cross-height candidate references.
    walls = files.get("walls")
    consensus = files.get("consensus")
    if isinstance(walls, list) and isinstance(consensus, dict):
        supports: dict[str, list[dict[str, object]]] = {}
        for shared in consensus.get("cross_band_consensus_candidates", []):
            for source in shared.get("source_candidates", []):
                if source.get("band") == "current":
                    supports.setdefault(str(source.get("id")), []).append(shared)
        for wall in walls:
            if "height_band_support_count" not in wall:
                matches = supports.get(str(wall.get("id")), [])
                strongest = max(matches, key=lambda item: item.get("support_count", 0), default=None)
                wall["height_band_support_count"] = strongest.get("support_count", 1) if strongest else 1

    payload["images"] = sorted(name for name, path in image_paths.items() if path.is_file())
    return payload


def make_handler(
    results_dir: Path, report_prefix: str | None = None
) -> type[BaseHTTPRequestHandler]:
    if report_prefix is None:
        image_paths = {name: results_dir / name for name in IMAGE_FILES}
    else:
        image_paths = {
            "plan.png": results_dir / f"{report_prefix}_step10_voxel2cm.png",
            "room_candidates.png": results_dir / f"{report_prefix}_room_candidates.png",
            "wall_candidates.png": results_dir / f"{report_prefix}_wall_candidates.png",
            "cross_height_wall_candidates.png": results_dir / f"{report_prefix}_wall_consensus.png",
        }

    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/":
                self._send_file(PAGE_PATH, "text/html; charset=utf-8")
            elif route == "/api/results":
                body = json.dumps(load_results(results_dir, report_prefix)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif route.startswith("/artifact/"):
                name = route.removeprefix("/artifact/")
                if name not in IMAGE_FILES:
                    self.send_error(404)
                    return
                path = image_paths[name]
                if not path.is_file():
                    self.send_error(404, f"Missing result image: {name}")
                    return
                self._send_file(path, "image/png")
            else:
                self.send_error(404)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[review] {self.address_string()} {format % args}")

    return ReviewHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open a local browser page to review one scan's generated results."
    )
    parser.add_argument("results_dir", nargs="?", type=Path, default=Path("runs/latest"))
    parser.add_argument(
        "--report-prefix",
        help="open existing reports named like PREFIX_room_candidates.json and PREFIX_step10_voxel2cm.png",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        parser.error(f"results directory does not exist: {results_dir}")
    if not PAGE_PATH.is_file():
        parser.error(f"review page is missing: {PAGE_PATH}")
    if args.report_prefix is not None and (
        Path(args.report_prefix).name != args.report_prefix or args.report_prefix in {".", ".."}
    ):
        parser.error("--report-prefix must be a filename prefix, not a path")

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(results_dir, args.report_prefix)
    )
    print(f"Reviewing results in {results_dir}")
    print(f"Open http://{args.host}:{server.server_port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
