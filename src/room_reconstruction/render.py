"""Rendering for the current point-cloud preview artifact."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from .geometry import estimate_floor_height


def render_point_cloud_preview(points: np.ndarray, path: str, title: str) -> None:
    """Save an x-z projection of the world-space points, labeled as a preview."""
    points = np.asarray(points)
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(points):
        image = axis.scatter(
            points[:, 0], points[:, 2], c=points[:, 1], s=0.35,
            cmap="viridis", linewidths=0, rasterized=True,
        )
        figure.colorbar(image, ax=axis, label="World height y (m)")
    else:
        axis.text(0.5, 0.5, "No valid points", transform=axis.transAxes,
                  ha="center", va="center")
    axis.set_title(f"{title}\nPOINT CLOUD PREVIEW — NOT A FLOOR PLAN")
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World z (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_wall_candidate_preview(
    points: np.ndarray, wall_candidates: list[dict[str, object]], path: str, title: str
) -> None:
    """Show wall-line candidates over the height slice used to fit them."""
    points = np.asarray(points)
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(points) == 0:
        axis.text(0.5, 0.5, "No valid points", transform=axis.transAxes,
                  ha="center", va="center")
        axis.set_title(f"{title}\nWALL CANDIDATES — NO DATA")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    floor_y = estimate_floor_height(points)
    slice_mask = (
        (points[:, 1] >= floor_y + 0.85) & (points[:, 1] <= floor_y + 1.55)
    )
    points = points[slice_mask]
    display_points = points[::max(1, len(points) // 150_000)]
    axis.scatter(display_points[:, 0], display_points[:, 2], s=0.25,
                 color="#87909a", alpha=0.45, linewidths=0, rasterized=True)
    for candidate in wall_candidates:
        start = candidate["start_xz_m"]
        end = candidate["end_xz_m"]
        axis.plot([start[0], end[0]], [start[1], end[1]], color="#d62728",
                  linewidth=2.0, alpha=0.9)
        support_count = int(candidate.get("height_band_support_count", 1))
        if support_count > 1:
            midpoint = (np.asarray(start) + np.asarray(end)) / 2
            axis.text(midpoint[0], midpoint[1], f"{support_count}/3",
                      color="#075b31", fontsize=8, ha="center", va="center",
                      bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    axis.set_title(
        f"{title} · height {floor_y + .85:.2f}–{floor_y + 1.55:.2f} m\n"
        "WALL CANDIDATES — labels show strongest subsection support; review required"
    )
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World z (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_wall_consensus_preview(
    points: np.ndarray,
    wall_candidates: list[dict[str, object]],
    path: str,
    title: str,
) -> None:
    """Show shared wall segments supported in multiple height bands."""
    points = np.asarray(points)
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(points) == 0:
        axis.text(0.5, 0.5, "No valid points", transform=axis.transAxes,
                  ha="center", va="center")
        axis.set_title(f"{title}\nCROSS-HEIGHT WALL SUPPORT — NO DATA")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return

    floor_y = estimate_floor_height(points)
    evidence = points[
        (points[:, 1] >= floor_y + 0.40) & (points[:, 1] <= floor_y + 2.20)
    ]
    display = evidence[::max(1, len(evidence) // 150_000)]
    axis.scatter(display[:, 0], display[:, 2], s=0.25, color="#87909a",
                 alpha=0.35, linewidths=0, rasterized=True)
    for candidate in wall_candidates:
        start = candidate["start_xz_m"]
        end = candidate["end_xz_m"]
        axis.plot([start[0], end[0]], [start[1], end[1]], color="#18864b",
                  linewidth=2.4, solid_capstyle="round")
        midpoint = (np.asarray(start) + np.asarray(end)) / 2
        axis.text(
            midpoint[0], midpoint[1], f"{candidate['support_count']}/3",
            color="#075b31", fontsize=8, ha="center", va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    axis.set_title(
        f"{title} · {len(wall_candidates)} shared segments\n"
        "Cross-height agreement candidates — review required"
    )
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World z (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_room_candidate_preview(
    points: np.ndarray, room_candidates: list[dict[str, object]], path: str, title: str,
    opening_candidates: list[dict[str, object]] | None = None,
    wall_gap_candidates: list[dict[str, object]] | None = None,
    wall_room_candidates: list[dict[str, object]] | None = None,
) -> None:
    """Show room outlines, room openings, and unverified wall gaps."""
    points = np.asarray(points)
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(points) == 0:
        axis.text(0.5, 0.5, "No valid points", transform=axis.transAxes,
                  ha="center", va="center")
        axis.set_title(f"{title}\nROOM CANDIDATES — NO DATA")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return
    floor_y = estimate_floor_height(points)
    in_slice = points[
        (points[:, 1] >= floor_y + 0.85) & (points[:, 1] <= floor_y + 1.55)
    ]
    display_points = in_slice[::max(1, len(in_slice) // 150_000)]
    axis.scatter(display_points[:, 0], display_points[:, 2], s=0.25,
                 color="#87909a", alpha=0.55, linewidths=0, rasterized=True)
    for candidate in room_candidates:
        boundary = candidate["boundary_xz_m"]
        if len(boundary) >= 4:
            axis.add_patch(Polygon(
                boundary, closed=True, fill=False, edgecolor="#e67e22",
                linewidth=2, linestyle="--",
            ))
        center = candidate["center_xz_m"]
        axis.text(center[0], center[1], candidate["id"], color="#9a4d00",
                  fontsize=8, ha="center", va="center",
                  bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    for candidate in wall_room_candidates or []:
        boundary = candidate["boundary_xz_m"]
        axis.add_patch(Polygon(
            boundary, closed=True, fill=False, edgecolor="#176b50",
            linewidth=2.5, linestyle="-",
        ))
        center = np.mean(np.asarray(boundary, dtype=np.float64), axis=0)
        axis.text(center[0], center[1], candidate["id"], color="#145b3e",
                  fontsize=8, ha="center", va="center",
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    if not wall_room_candidates:
        axis.text(
            0.02, 0.98,
            "No wall outline met the 70% per-side scan-support threshold.",
            transform=axis.transAxes, ha="left", va="top", color="#145b3e",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#176b50"},
        )
    for opening in opening_candidates or []:
        endpoints = opening["gap_endpoints_xz_m"]
        axis.plot([endpoints[0][0], endpoints[1][0]],
                  [endpoints[0][1], endpoints[1][1]],
                  color="#d627a8", linewidth=4, solid_capstyle="round")
        center = opening["center_xz_m"]
        axis.text(center[0], center[1], opening["id"], color="#9c146f",
                  fontsize=8, ha="center", va="bottom",
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    for gap in wall_gap_candidates or []:
        endpoints = gap["gap_endpoints_xz_m"]
        axis.plot([endpoints[0][0], endpoints[1][0]],
                  [endpoints[0][1], endpoints[1][1]],
                  color="#177c9c", linewidth=4, linestyle="--",
                  solid_capstyle="round")
        center = gap["center_xz_m"]
        axis.text(center[0], center[1], f"{gap['observed_gap_m']:.2f} m",
                  color="#07556b", fontsize=8, ha="center", va="top",
                  bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    axis.set_title(f"{title}\nROOM OUTLINES, OPENINGS AND WALL GAPS — REVIEW REQUIRED")
    axis.set_xlabel("World x (m)")
    axis.set_ylabel("World z (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.18)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
