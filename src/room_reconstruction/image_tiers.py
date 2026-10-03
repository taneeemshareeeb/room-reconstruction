"""Visual overlap processing for photo and video inputs.

This produces an image-space panorama and overlap diagnostics. Monocular image
inputs have no reliable scale or camera poses, so this module deliberately
does not report metric room geometry.
"""

from __future__ import annotations

import json
from io import BytesIO
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".zip"}
MAX_PHOTO_COUNT = 120
MAX_VIDEO_FRAMES = 60
MAX_MOSAIC_PIXELS = 12_000_000


def _collect_inputs(capture: Path, tier: str) -> list[Path]:
    allowed = IMAGE_SUFFIXES if tier == "photos" else VIDEO_SUFFIXES
    if capture.is_file():
        if capture.suffix.lower() not in allowed:
            raise ValueError(f"{capture.name} is not a supported {tier} input")
        return [capture]
    if not capture.is_dir():
        raise FileNotFoundError(f"capture path does not exist: {capture}")
    paths = sorted(
        item for item in capture.rglob("*")
        if item.is_file() and item.suffix.lower() in allowed
    )
    if not paths:
        raise ValueError(f"no supported {tier} inputs found in {capture}")
    return paths[:MAX_PHOTO_COUNT if tier == "photos" else 1]


def _read_photos(paths: list[Path], max_width: int = 1200) -> tuple[list[np.ndarray], list[str]]:
    images, names = [], []
    errors = []
    for path in paths:
        try:
            with Image.open(path) as source:
                rgb = source.convert("RGB")
                if rgb.width > max_width:
                    height = round(rgb.height * max_width / rgb.width)
                    rgb = rgb.resize((max_width, height), Image.Resampling.LANCZOS)
                images.append(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
                names.append(path.name)
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    if not images:
        raise ValueError("no images could be decoded; " + "; ".join(errors[:3]))
    return images, names


def _read_video(path: Path, sample_seconds: float) -> tuple[list[np.ndarray], list[str], float, float]:
    if path.suffix.lower() == ".zip":
        return _read_video_from_zip(path, sample_seconds)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("video does not report usable frame rate and length metadata")
    interval = max(1, round(fps * sample_seconds), (frame_count - 1) // (MAX_VIDEO_FRAMES - 1))
    frames, names = [], []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % interval == 0:
                height, width = frame.shape[:2]
                if width > 1200:
                    frame = cv2.resize(frame, (1200, round(height * 1200 / width)), interpolation=cv2.INTER_AREA)
                frames.append(frame)
                names.append(f"{path.name}@{index / fps:.2f}s")
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contained no decodable frames: {path}")
    return frames, names, fps, interval / fps


def _read_video_from_zip(
    path: Path, sample_seconds: float,
) -> tuple[list[np.ndarray], list[str], float, float]:
    """Decode an rgb.mp4 member from seekable memory without extracting it."""
    import av

    with zipfile.ZipFile(path) as archive:
        videos = [item for item in archive.infolist() if Path(item.filename).name.lower() == "rgb.mp4"]
        if len(videos) != 1:
            raise ValueError(f"expected one rgb.mp4 in {path.name}, found {len(videos)}")
        member = videos[0]
        if member.file_size > 1_000_000_000:
            raise ValueError("rgb.mp4 is too large for bounded in-memory video decoding")
        video_bytes = archive.read(member)
    try:
        container = av.open(BytesIO(video_bytes))
    except av.FFmpegError as exc:
        raise ValueError(f"could not open rgb.mp4 from ZIP: {exc}") from exc
    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError("rgb.mp4 contains no video stream")
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("rgb.mp4 does not report a usable frame rate")
        duration_s = (
            float(stream.duration * stream.time_base)
            if stream.duration and stream.time_base
            else float(container.duration / av.time_base) if container.duration else 0.0
        )
        frame_count = int(stream.frames or round(duration_s * fps))
        interval = max(
            1, round(fps * sample_seconds),
            (frame_count - 1) // (MAX_VIDEO_FRAMES - 1) if frame_count > 1 else 1,
        )
        frames, names = [], []
        frame_index = 0
        for decoded in container.decode(stream):
            if frame_index % interval == 0:
                frame = decoded.to_ndarray(format="bgr24")
                height, width = frame.shape[:2]
                if width > 1200:
                    frame = cv2.resize(
                        frame, (1200, round(height * 1200 / width)), interpolation=cv2.INTER_AREA
                    )
                frames.append(frame)
                names.append(f"rgb.mp4@{frame_index / fps:.2f}s")
            frame_index += 1
    except av.FFmpegError as exc:
        raise ValueError(f"could not decode rgb.mp4 from ZIP: {exc}") from exc
    finally:
        container.close()
    if not frames:
        raise ValueError("no sampled frames could be decoded from rgb.mp4")
    return frames, names, fps, interval / fps


def _estimate_mosaic(
    images: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, object]], int, str]:
    detector = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    keypoints_descriptors = [detector.detectAndCompute(image, None) for image in images]
    transforms = [np.eye(3, dtype=np.float64)]
    pairs: list[dict[str, object]] = []
    accepted = 0
    for index in range(1, len(images)):
        previous_kp, previous_desc = keypoints_descriptors[index - 1]
        current_kp, current_desc = keypoints_descriptors[index]
        record: dict[str, object] = {
            "from_image_index": index,
            "to_image_index": index - 1,
            "status": "insufficient_feature_matches",
            "keypoints_from": len(current_kp),
            "keypoints_to": len(previous_kp),
            "good_matches": 0,
            "ransac_inliers": 0,
            "inlier_fraction": 0.0,
        }
        if current_desc is not None and previous_desc is not None:
            raw = matcher.knnMatch(current_desc, previous_desc, k=2)
            good = [a for pair in raw if len(pair) == 2 for a, b in [pair] if a.distance < 0.72 * b.distance]
            record["good_matches"] = len(good)
            if len(good) >= 12:
                source = np.float32([current_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                target = np.float32([previous_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
                inliers = int(mask.sum()) if mask is not None else 0
                fraction = inliers / len(good)
                record["ransac_inliers"] = inliers
                record["inlier_fraction"] = round(fraction, 3)
                if homography is not None and inliers >= 10 and fraction >= 0.15:
                    candidate = transforms[-1] @ homography
                    h, w = images[index].shape[:2]
                    corners = cv2.perspectiveTransform(
                        np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), candidate
                    )[0]
                    if np.isfinite(corners).all() and np.max(np.abs(corners)) < 20000:
                        transforms.append(candidate)
                        accepted += 1
                        record["status"] = "homography_candidate"
                    else:
                        transforms.append(transforms[-1].copy())
                        record["status"] = "rejected_unstable_homography"
                else:
                    transforms.append(transforms[-1].copy())
            else:
                transforms.append(transforms[-1].copy())
        else:
            transforms.append(transforms[-1].copy())
        pairs.append(record)

    corners_all = []
    for image, transform in zip(images, transforms):
        h, w = image.shape[:2]
        corners_all.append(cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), transform)[0])
    all_corners = np.concatenate(corners_all)
    minimum, maximum = np.floor(all_corners.min(axis=0)), np.ceil(all_corners.max(axis=0))
    width, height = (maximum - minimum).astype(int)
    required_registrations = max(1, int(np.ceil((len(images) - 1) * 0.2)))
    if (
        accepted < required_registrations
        or width < 1 or height < 1
        or width * height > MAX_MOSAIC_PIXELS
    ):
        columns = 4
        tile_w, tile_h, title_h, margin = 320, 240, 30, 12
        rows = (len(images) + columns - 1) // columns
        sheet = np.full(
            (margin + rows * (tile_h + title_h + margin), margin + columns * (tile_w + margin), 3),
            245, dtype=np.uint8,
        )
        for index, image in enumerate(images):
            row, col = divmod(index, columns)
            x, y = margin + col * (tile_w + margin), margin + row * (tile_h + title_h + margin)
            thumb = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            sheet[y:y + tile_h, x:x + tile_w] = thumb
            cv2.putText(sheet, f"Image {index + 1}", (x, y + tile_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        return sheet, pairs, accepted, "contact_sheet"
    translate = np.array([[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64)
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for image, transform in zip(images, transforms):
        warped = cv2.warpPerspective(image, translate @ transform, (width, height))
        mask = cv2.warpPerspective(np.ones(image.shape[:2], np.float32), translate @ transform, (width, height))
        accumulator += warped.astype(np.float32) * mask[:, :, None]
        weights += mask
    mosaic = np.divide(accumulator, np.maximum(weights[:, :, None], 1.0), out=np.zeros_like(accumulator), where=weights[:, :, None] > 0)
    return np.clip(mosaic, 0, 255).astype(np.uint8), pairs, accepted, "mosaic_candidate"


def process_image_tier(
    capture: str | Path,
    *,
    tier: str,
    output_dir: str | Path,
    sample_seconds: float = 1.0,
) -> dict[str, object]:
    """Process photo/video inputs into a visual overlap map and honest report."""
    if tier not in {"photos", "video"}:
        raise ValueError("image tier must be 'photos' or 'video'")
    if sample_seconds <= 0 or not np.isfinite(sample_seconds):
        raise ValueError("sample_seconds must be positive and finite")
    input_path = Path(capture)
    try:
        paths = _collect_inputs(input_path, tier)
        frame_rate = None
        actual_sample_seconds = None
        if tier == "photos":
            images, names = _read_photos(paths)
        else:
            images, names, frame_rate, actual_sample_seconds = _read_video(paths[0], sample_seconds)
        if len(images) < 2:
            raise ValueError(f"{tier} tier needs at least two usable images or sampled video frames")
        mosaic, pairs, accepted, preview_kind = _estimate_mosaic(images)
    except cv2.error as exc:
        raise ValueError(f"image registration failed: {exc}") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preview_path = output / "plan.png"
    banner_height = 52
    labeled = np.zeros((mosaic.shape[0] + banner_height, mosaic.shape[1], 3), dtype=np.uint8)
    labeled[banner_height:] = mosaic
    cv2.putText(
        labeled,
        (f"{tier.upper()} MOSAIC CANDIDATE - NOT A METRIC FLOOR PLAN"
         if preview_kind == "mosaic_candidate"
         else f"{tier.upper()} CONTACT SHEET - NO STABLE GLOBAL MOSAIC FOUND"),
        (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    if not cv2.imwrite(str(preview_path), labeled):
        raise OSError(f"could not write preview image: {preview_path}")
    report: dict[str, object] = {
        "tier": tier,
        "input": str(input_path),
        "image_count": len(images),
        "image_names": names,
        "adjacent_pair_matches": pairs,
        "registered_adjacent_pairs": accepted,
        "minimum_registrations_for_mosaic": max(1, int(np.ceil((len(images) - 1) * 0.2))),
        "registered_pair_fraction": round(accepted / max(1, len(images) - 1), 3),
        "preview_kind": preview_kind,
        "mosaic_size_px": [int(mosaic.shape[1]), int(mosaic.shape[0])],
        "video_fps": frame_rate,
        "sample_seconds": sample_seconds if tier == "video" else None,
        "actual_sampling_interval_s": actual_sample_seconds,
        "video_frame_limit": MAX_VIDEO_FRAMES if tier == "video" else None,
        "video_sampling_note": (
            f"At most {MAX_VIDEO_FRAMES} frames are sampled across the video; the effective interval "
            f"is {actual_sample_seconds:.2f} seconds."
            if tier == "video" else None
        ),
        "metric_geometry_status": "insufficient_data",
        "metric_geometry_reason": (
            "Monocular images do not provide reliable metric scale or camera poses. "
            "The visual mosaic and feature matches are overlap diagnostics, not a 3D room reconstruction."
        ),
    }
    (output / "image_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def visual_plan_document(
    *, tier: str, capture: str, image_analysis: dict[str, object]
) -> dict[str, object]:
    """Create a schema-compatible result without inventing metric geometry."""
    return {
        "schema_version": "0.7.0",
        "status": "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture": {"id": capture, "tier": tier},
        "processing": {
            "frame_step": 1,
            "voxel_size_m": 0.02,
            "point_count": 0,
            "pass_split": None,
        },
        "rooms": [],
        "wall_candidates": [],
        "cross_height_wall_candidates": [],
        "room_candidates": [],
        "opening_candidates": [],
        "wall_gap_candidates": [],
        "wall_room_candidates": [],
        "partial_wall_room_candidates": [],
        "floor_height_candidate": None,
        "ceiling_height_candidates": [],
        "ground_truth_comparison": None,
        "room_connections": [],
        "scope_line_items": [],
        "capabilities": {
            "point_cloud": "insufficient_data",
            "visual_overlap": "candidate_available",
            "room_segmentation": "insufficient_data",
            "walls": "insufficient_data",
            "openings": "insufficient_data",
            "floor_area": "insufficient_data",
            "photos_tier": "candidate_available" if tier == "photos" else "insufficient_data",
            "video_tier": "candidate_available" if tier == "video" else "insufficient_data",
        },
        "artifacts": {"preview": "plan.png", "visual_analysis": "image_analysis.json"},
        "limitations": [
            str(image_analysis["metric_geometry_reason"]),
            "Image registration uses adjacent-frame feature homographies; accumulated mosaic drift is possible.",
            "No calibrated camera trajectory, room scale, or validated wall measurements are available for this tier.",
        ],
    }
