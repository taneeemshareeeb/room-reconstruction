#!/usr/bin/env python3
"""Serve a local, read-only review page for one reconstruction result folder."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = PROJECT_ROOT / "scripts" / "review_ui" / "index.html"
REVIEW_UI_DIR = PAGE_PATH.parent
DEMO_RESULTS = {
    "single_room": PROJECT_ROOT / "runs" / "single_room_current",
    "floor_only": PROJECT_ROOT / "runs" / "single_scan_floor_only",
    "with_ceiling": PROJECT_ROOT / "runs" / "single_scan_with_ceiling",
}
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
LIDAR_UPLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "phone_uploads"
MAX_LIDAR_UPLOAD_BYTES = 600 * 1024 * 1024
MAX_LIDAR_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_LIDAR_ARCHIVE_ENTRIES = 100_000


def _validate_lidar_archive(archive_path: Path) -> dict[str, int]:
    """Check the archive structure consumed by load_archive_point_cloud."""
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_LIDAR_ARCHIVE_ENTRIES:
                raise ValueError("archive contains too many files")
            if sum(entry.file_size for entry in entries) > MAX_LIDAR_UNCOMPRESSED_BYTES:
                raise ValueError("archive expands beyond the 4 GiB safety limit")
            for entry in entries:
                normalized = entry.filename.replace("\\", "/")
                path = PurePosixPath(normalized)
                if path.is_absolute() or ".." in path.parts or re.match(r"^[a-zA-Z]:", normalized):
                    raise ValueError("archive contains an unsafe member path")

            odometry_paths = [
                entry.filename for entry in entries
                if entry.filename.endswith("/odometry.csv")
            ]
            if len(odometry_paths) != 1:
                raise ValueError(f"expected one nested odometry.csv, found {len(odometry_paths)}")
            odometry_path = odometry_paths[0]
            prefix = PurePosixPath(odometry_path).parent.as_posix()
            members = {entry.filename for entry in entries}
            with archive.open(odometry_path) as binary_stream:
                import io

                with io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream, skipinitialspace=True)
                    required = {"frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"}
                    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                        raise ValueError("odometry.csv is missing required pose or camera-intrinsic columns")
                    frame_ids = []
                    for row in reader:
                        try:
                            frame_id = int(row["frame"])
                        except (TypeError, ValueError) as exc:
                            raise ValueError("odometry.csv contains an invalid frame number") from exc
                        if frame_id < 0 or frame_id in frame_ids:
                            raise ValueError("odometry.csv frame numbers must be unique and non-negative")
                        frame_ids.append(frame_id)
            if not frame_ids:
                raise ValueError("odometry.csv contains no frames")
            missing = [
                f"{kind}/{frame_id:06d}.png"
                for frame_id in frame_ids[: min(len(frame_ids), 5)]
                for kind in ("depth", "confidence")
                if f"{prefix}/{kind}/{frame_id:06d}.png" not in members
            ]
            if missing:
                raise ValueError("archive is missing required depth/confidence frames: " + ", ".join(missing))
            return {"frame_count": len(frame_ids), "entry_count": len(entries)}
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
        raise ValueError(f"could not read LiDAR ZIP archive: {exc}") from exc


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
    results_dir: Path, report_prefix: str | None = None, *, app_mode: bool = False,
    camera_mode: bool = False,
) -> type[BaseHTTPRequestHandler]:
    job_state: dict[str, object] = {"status": "idle"}
    job_lock = threading.Lock()
    current_results_dir = [results_dir]
    try:
        default_output_dir = str(results_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        default_output_dir = "runs/latest"
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
            query = parse_qs(urlsplit(self.path).query)
            if route == "/":
                body = PAGE_PATH.read_bytes().replace(
                    b"__APP_MODE__", str(app_mode).lower().encode("ascii")
                ).replace(
                    b"__DEFAULT_OUTPUT_DIR__",
                    html.escape(default_output_dir, quote=True).encode("utf-8"),
                ).replace(
                    b"__CAMERA_MODE__", str(camera_mode).lower().encode("ascii")
                )
                self._send_bytes(body, "text/html; charset=utf-8")
            elif route == "/api/results":
                requested_demo = (query.get("demo") or [None])[0]
                if requested_demo in DEMO_RESULTS:
                    current_results_dir[0] = DEMO_RESULTS[requested_demo]
                body = json.dumps(load_results(current_results_dir[0], report_prefix)).encode("utf-8")
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
            elif route == "/api/capture-sessions" and camera_mode:
                self._send_json({"sessions": _load_capture_sessions()})
            elif route.startswith("/api/capture-sessions/") and camera_mode:
                parts = route.strip("/").split("/")
                if (
                    len(parts) != 4 or parts[0:2] != ["api", "capture-sessions"]
                    or not re.fullmatch(r"[a-f0-9]{32}", parts[2])
                    or not re.fullmatch(r"frame_\d{3}\.(?:jpg|png)", parts[3])
                ):
                    self.send_error(404)
                    return
                path = CAMERA_SESSIONS_DIR / parts[2] / parts[3]
                self._send_file(path, "image/png" if path.suffix == ".png" else "image/jpeg")
            elif route == "/manifest.webmanifest":
                path = REVIEW_UI_DIR / "manifest.webmanifest"
                self._send_file(path, "application/manifest+json")
            elif route in {"/icon-192.png", "/apple-touch-icon.png"}:
                path = REVIEW_UI_DIR / "icon-192.png"
                self._send_file(path, "image/png")
            elif route == "/icon-512.png":
                path = REVIEW_UI_DIR / "icon-512.png"
                self._send_file(path, "image/png")
            elif route.startswith("/artifact/"):
                name = route.removeprefix("/artifact/")
                if name not in IMAGE_FILES:
                    self.send_error(404)
                    return
                path = (
                    current_results_dir[0] / name
                    if app_mode and report_prefix is None
                    else image_paths[name]
                )
                if not path.is_file():
                    self.send_error(404, f"Missing result image: {name}")
                    return
                self._send_file(path, "image/png")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            route = urlsplit(self.path).path
            if route == "/api/capture-sessions" and camera_mode:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > MAX_CAPTURE_REQUEST_BYTES:
                        raise ValueError("request body must be between 1 byte and 50 MiB")
                    if self.headers.get_content_type() != "application/json":
                        raise ValueError("Content-Type must be application/json")
                    payload = json.loads(self.rfile.read(length))
                    session = _store_camera_session(payload)
                except (ValueError, TypeError, json.JSONDecodeError, OverflowError) as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                except OSError as exc:
                    self._send_json({"error": f"could not store camera session: {exc}"}, status=500)
                    return
                self._send_json(session, status=201)
                return
            if route != "/api/process" or not app_mode:
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
                output_text = payload.get("output_dir")
                if not isinstance(output_text, str) or not output_text.strip():
                    raise ValueError("output folder is required")
                requested_output = Path(output_text.strip()).expanduser()
                output_dir = (
                    PROJECT_ROOT / requested_output
                    if not requested_output.is_absolute()
                    else requested_output
                ).resolve()
                output_dir.relative_to((PROJECT_ROOT / "runs").resolve())
                if output_dir == (PROJECT_ROOT / "runs").resolve():
                    raise ValueError("choose a result subfolder inside runs")
                if capture == output_dir or output_dir in capture.parents or capture in output_dir.parents:
                    raise ValueError("capture path and output folder cannot overlap")
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
                job_state.update({
                    "status": "running",
                    "capture": str(capture.relative_to(PROJECT_ROOT)),
                    "tier": tier,
                    "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                })

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
                        if completed.returncode == 0:
                            current_results_dir[0] = output_dir
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
    parser.add_argument("--camera-capture", action="store_true", help="enable browser camera session uploads")
    parser.add_argument("--allow-lan", action="store_true", help="allow trusted LAN access; requires HTTPS certificate files")
    parser.add_argument("--tls-cert", type=Path, help="TLS certificate PEM file for HTTPS")
    parser.add_argument("--tls-key", type=Path, help="TLS private key PEM file for HTTPS")
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

    if args.camera_capture and not args.app:
        parser.error("--camera-capture requires --app")
    if args.allow_lan and (not args.app or not args.camera_capture):
        parser.error("--allow-lan requires --app and --camera-capture")
    if args.allow_lan and args.host not in {"0.0.0.0", "::"}:
        parser.error("--allow-lan requires --host 0.0.0.0 or --host ::")
    if args.app and args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_lan:
        parser.error("non-local app binding requires --allow-lan")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be supplied together")
    if args.allow_lan and not (args.tls_cert and args.tls_key):
        parser.error("--allow-lan requires --tls-cert and --tls-key for iPhone camera access")
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(
            results_dir, args.report_prefix, app_mode=args.app, camera_mode=args.camera_capture
        )
    )
    if args.tls_cert and args.tls_key:
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = tls_context.wrap_socket(server.socket, server_side=True)
    print(f"Room Reconstruction app writing to {results_dir}" if args.app else f"Reviewing results in {results_dir}")
    protocol = "https" if args.tls_cert else "http"
    display_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Open {protocol}://{display_host}:{server.server_port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
