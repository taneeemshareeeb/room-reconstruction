"""Conservative floor and ceiling height candidates from separate scan passes."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from .geometry import estimate_floor_height


def detect_vertical_surface_candidates(
    floor_pass_points: np.ndarray,
    ceiling_pass_points: np.ndarray | None,
    *,
    floor_reference_pass: str = "A",
    bin_size_m: float = 0.03,
    minimum_ceiling_height_m: float = 2.0,
    maximum_ceiling_height_m: float = 4.2,
) -> dict[str, object]:
    """Return a floor mode and distinct upper height peaks without room claims.

    Peak widths describe the observed histogram shape; they are not calibrated
    confidence intervals. Ceiling candidates from a pass are not associated
    with individual rooms or classified as structural ceilings.
    """
    floor_cloud = np.asarray(floor_pass_points, dtype=np.float64)
    if floor_cloud.ndim != 2 or floor_cloud.shape[1] != 3 or len(floor_cloud) == 0:
        raise ValueError("floor-pass points must be a non-empty N x 3 array")
    if not np.isfinite(floor_cloud).all():
        raise ValueError("floor-pass points must be finite")
    if floor_reference_pass not in {"A", "unsegmented"}:
        raise ValueError("floor_reference_pass must be 'A' or 'unsegmented'")
    if (
        not np.isfinite([bin_size_m, minimum_ceiling_height_m, maximum_ceiling_height_m]).all()
        or bin_size_m <= 0
        or minimum_ceiling_height_m <= 0
        or maximum_ceiling_height_m <= minimum_ceiling_height_m
    ):
        raise ValueError("height histogram settings are invalid")

    floor_y = estimate_floor_height(floor_cloud)
    floor_band = floor_cloud[np.abs(floor_cloud[:, 1] - floor_y) <= 0.12, 1]
    floor_half_width = (
        float(np.percentile(np.abs(floor_band - floor_y), 90)) if len(floor_band) else None
    )
    floor_candidate = {
        "elevation_world_y_m": round(float(floor_y), 3),
        "observed_half_width_m": round(floor_half_width, 3) if floor_half_width is not None else None,
        "supporting_points_in_0_24m_band": int(len(floor_band)),
        "reference_pass": floor_reference_pass,
        "status": "candidate_unverified",
        "reason": (
            "Strong low-height histogram mode in the downward-looking pass; floor-plane quality "
            "and room assignment remain unverified."
            if floor_reference_pass == "A"
            else "Strong low-height histogram mode in the unsegmented capture; floor-plane quality "
            "and room assignment remain unverified."
        ),
    }

    ceiling_candidates = []
    if ceiling_pass_points is not None:
        upper_cloud = np.asarray(ceiling_pass_points, dtype=np.float64)
        if upper_cloud.ndim != 2 or upper_cloud.shape[1] != 3:
            raise ValueError("ceiling-pass points must have shape N x 3")
        if not np.isfinite(upper_cloud).all():
            raise ValueError("ceiling-pass points must be finite")
        low_y = floor_y + minimum_ceiling_height_m
        high_y = floor_y + maximum_ceiling_height_m
        upper_y = upper_cloud[
            (upper_cloud[:, 1] >= low_y) & (upper_cloud[:, 1] <= high_y), 1
        ]
        if len(upper_y) >= 20:
            edges = np.arange(low_y, high_y + bin_size_m, bin_size_m)
            counts, edges = np.histogram(upper_y, bins=edges)
            smooth = gaussian_filter1d(counts.astype(np.float64), sigma=1.0)
            min_prominence = max(3.0, float(smooth.max()) * 0.05)
            peaks, _ = find_peaks(
                smooth,
                prominence=min_prominence,
                distance=max(1, int(round(0.25 / bin_size_m))),
            )
            widths, _, _, _ = peak_widths(smooth, peaks, rel_height=0.5)
            ranked = sorted(peaks, key=lambda index: smooth[index], reverse=True)[:8]
            for peak_index in ranked:
                peak_position = int(np.flatnonzero(peaks == peak_index)[0])
                elevation = float((edges[peak_index] + edges[peak_index + 1]) / 2)
                half_width = float(widths[peak_position] * bin_size_m / 2)
                ceiling_candidates.append({
                    "id": f"ceiling_height_candidate_{len(ceiling_candidates) + 1:02d}",
                    "elevation_world_y_m": round(elevation, 3),
                    "height_above_floor_m": round(elevation - floor_y, 3),
                    "observed_half_width_m": round(half_width, 3),
                    "peak_supporting_points": int(counts[peak_index]),
                    "fraction_of_upper_pass_slice": round(float(counts[peak_index] / len(upper_y)), 4),
                    "reference_pass": "B",
                    "status": "candidate_unverified",
                    "reason": (
                        "Height-histogram peak in the upward-looking pass; it may be a ceiling, "
                        "beam, header, or another height level and is not linked to a room."
                    ),
                })

    return {
        "floor_height_candidate": floor_candidate,
        "ceiling_height_candidates": ceiling_candidates,
        "ceiling_candidate_status": (
            "candidate_available" if ceiling_candidates else "insufficient_data"
        ),
    }
