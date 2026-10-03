#!/usr/bin/env python3
"""Serve a local, read-only review page for one reconstruction result folder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
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
    "wall_gaps": "wall_gap_candidates.json",
    "wall_rooms": "wall_room_candidates.json",
    "partial_wall_rooms": "partial_wall_room_candidates.json",
    "consensus": "cross_height_wall_candidates.json",
    "image_analysis": "image_analysis.json",
}
IMAGE_FILES = {
    "plan.png",
    "room_candidates.png",
    "room_layout_draft.png",
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
            "wall_rooms": None,
            "partial_wall_rooms": None,
        }
        image_paths = {
            "plan.png": results_dir / f"{report_prefix}_step10_voxel2cm.png",
            "room_candidates.png": results_dir / f"{report_prefix}_room_candidates.png",
            "wall_candidates.png": results_dir / f"{report_prefix}_wall_candidates.png",
            "cross_height_wall_candidates.png": results_dir / f"{report_prefix}_wall_consensus.png",
        }

    for key, path in json_paths.items():
        if path is None:
            if key in {"wall_rooms", "partial_wall_rooms"}:
                files[key] = []
                continue
            files[key] = {
                "capture": {"id": report_prefix},
                "processing": {"frame_step": 10, "voxel_size_m": 0.02, "point_count": None},
            }
            continue
        if not path.is_file():
            files[key] = None
            plan = files.get("plan")
            visual_tier = isinstance(plan, dict) and (plan.get("capture") or {}).get("tier") in {"photos", "video"}
            if key == "image_analysis" or (visual_tier and key in {"rooms", "walls", "openings", "wall_gaps", "wall_rooms", "partial_wall_rooms", "consensus"}):
                files[key] = [] if key != "image_analysis" else None
            else:
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

    plan = files.get("plan")
    visual_tier = isinstance(plan, dict) and (plan.get("capture") or {}).get("tier") in {"photos", "video"}
    if visual_tier:
        image_paths = {"plan.png": image_paths["plan.png"]}
    payload["images"] = sorted(name for name, path in image_paths.items() if path.is_file())
    return payload


def make_handler(
    results_dir: Path, report_prefix: str | None = None, *, app_mode: bool = False
) -> type[BaseHTTPRequestHandler]:
    job_state: dict[str, object] = {"status": "idle"}
    job_lock = threading.Lock()
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
                body = PAGE_PATH.read_bytes().replace(b"__APP_MODE__", str(app_mode).lower().encode("ascii"))
                self._send_bytes(body, "text/html; charset=utf-8")
            elif route == "/api/results":
                body = json.dumps(load_results(results_dir, report_prefix)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/job":
                with job_lock:
                    body = json.dumps(job_state).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
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

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/process" or not app_mode:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_384:
                    raise ValueError("request body must be between 1 and 16384 bytes")
                payload = json.loads(self.rfile.read(length))
                tier = payload.get("tier")
                if tier not in {"lidar", "photos", "video"}:
                    raise ValueError("tier must be lidar, photos, or video")
                capture_text = payload.get("capture")
                if not isinstance(capture_text, str) or not capture_text.strip():
                    raise ValueError("capture path is required")
                raw_capture = Path(capture_text.strip()).expanduser()
                capture = (PROJECT_ROOT / raw_capture).resolve() if not raw_capture.is_absolute() else raw_capture.resolve()
                capture.relative_to((PROJECT_ROOT / "data" / "raw").resolve())
                if not capture.exists():
                    raise ValueError("capture path does not exist inside the project")
                output_dir = results_dir.resolve()
                output_dir.relative_to(PROJECT_ROOT)
                if capture == output_dir or output_dir in capture.parents or capture in output_dir.parents:
                    raise ValueError("capture path cannot be the output folder or one of its parents")
                frame_step = int(payload.get("frame_step", 10))
                voxel_size = float(payload.get("voxel_size_m", 0.02))
                sample_seconds = float(payload.get("sample_seconds", 1.0))
                if frame_step < 1 or voxel_size <= 0 or sample_seconds <= 0:
                    raise ValueError("frame step, voxel size, and sample interval must be positive")
            except (ValueError, TypeError, json.JSONDecodeError, OverflowError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            with job_lock:
                if job_state.get("status") == "running":
                    self._send_json({"error": "a capture is already processing"}, status=409)
                    return
                job_state.clear()
                job_state.update({"status": "running", "capture": str(capture.relative_to(PROJECT_ROOT)), "tier": tier})

            command = [
                sys.executable, str(PROJECT_ROOT / "run.py"), str(capture),
                "--tier", tier, "--output-dir", str(output_dir),
                "--frame-step", str(frame_step), "--voxel-size", str(voxel_size),
                "--sample-seconds", str(sample_seconds),
            ]

            def run_job() -> None:
                try:
                    completed = subprocess.run(
                        command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3600
                    )
                    with job_lock:
                        job_state.clear()
                        job_state.update({
                            "status": "complete" if completed.returncode == 0 else "failed",
                            "return_code": completed.returncode,
                            "stdout": completed.stdout[-8000:],
                            "stderr": completed.stderr[-4000:],
                        })
                except (OSError, subprocess.TimeoutExpired) as exc:
                    with job_lock:
                        job_state.clear()
                        job_state.update({"status": "failed", "stderr": str(exc)})

            threading.Thread(target=run_job, daemon=True).start()
            self._send_json({"status": "running"}, status=202)

        def _send_json(self, value: dict[str, object], *, status: int = 200) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str, *, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(404)
                return
            self._send_bytes(body, content_type)

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
    parser.add_argument("--app", action="store_true", help="enable the local capture processing form")
    args = parser.parse_args(argv)
    results_dir = args.results_dir.resolve()
    if args.app:
        try:
            results_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            parser.error("app output folder must be inside the project")
        results_dir.mkdir(parents=True, exist_ok=True)
    if not results_dir.is_dir():
        parser.error(f"results directory does not exist: {results_dir}")
    if not PAGE_PATH.is_file():
        parser.error(f"review page is missing: {PAGE_PATH}")
    if args.report_prefix is not None and (
        Path(args.report_prefix).name != args.report_prefix or args.report_prefix in {".", ".."}
    ):
        parser.error("--report-prefix must be a filename prefix, not a path")

    if args.app and args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the processing app can only bind to localhost")
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(results_dir, args.report_prefix, app_mode=args.app)
    )
    print(f"Room Reconstruction app writing to {results_dir}" if args.app else f"Reviewing results in {results_dir}")
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
