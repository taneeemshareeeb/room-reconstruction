"""Conservative enclosed-region candidates from a horizontal point-cloud slice."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import estimate_floor_height


def _outer_boundary(mask: np.ndarray) -> list[list[int]]:
    """Trace the largest edge loop around a 4-connected raster region."""
    rows, columns = mask.shape
    edges = set()
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if column == 0 or not mask[row, column - 1]:
            edges.add(((row, column), (row + 1, column)))
        if row == rows - 1 or not mask[row + 1, column]:
            edges.add(((row + 1, column), (row + 1, column + 1)))
        if column == columns - 1 or not mask[row, column + 1]:
            edges.add(((row + 1, column + 1), (row, column + 1)))
        if row == 0 or not mask[row - 1, column]:
            edges.add(((row, column + 1), (row, column)))

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
    loops = []
    unused = set(edges)
    while unused:
        first = min(unused)
        start, current = first
        path = [start, current]
        unused.remove(first)
        while current != start:
            options = [end for end in outgoing.get(current, ()) if (current, end) in unused]
            if not options:
                break
            previous = path[-2]
            direction = np.subtract(current, previous)
            def turn(end):
                next_direction = np.subtract(end, current)
                return np.arctan2(
                    direction[0] * next_direction[1] - direction[1] * next_direction[0],
                    np.dot(direction, next_direction),
                )
            next_point = max(options, key=turn)
            unused.remove((current, next_point))
            path.append(next_point)
            current = next_point
        if len(path) >= 4 and path[-1] == start:
            # Remove collinear grid steps while retaining the closed outline.
            simplified = []
            ring = path[:-1]
            for index, point in enumerate(ring):
                before = np.asarray(ring[index - 1])
                here = np.asarray(point)
                after = np.asarray(ring[(index + 1) % len(ring)])
                incoming = here - before
                outgoing_vector = after - here
                cross = incoming[0] * outgoing_vector[1] - incoming[1] * outgoing_vector[0]
                if cross != 0:
                    simplified.append(point)
            if len(simplified) >= 3:
                simplified.append(simplified[0])
                loops.append(simplified)
    if not loops:
        return []

    def area(loop):
        coordinates = np.asarray(loop[:-1], dtype=np.float64)
        return abs(float(np.dot(coordinates[:, 0], np.roll(coordinates[:, 1], -1))
                           - np.dot(coordinates[:, 1], np.roll(coordinates[:, 0], -1)))) / 2
    return max(loops, key=area)


def detect_room_candidates(
    points: np.ndarray,
    *,
    grid_rotation_deg: float = 0.0,
    grid_size_m: float = 0.08,
    slice_min_height_m: float = 0.85,
    slice_max_height_m: float = 1.55,
    gap_closing_cells: int = 5,
    min_area_m2: float = 2.5,
    max_area_m2: float = 40.0,
    min_dimension_m: float = 1.5,
) -> list[dict[str, object]]:
    """Return candidate enclosed regions; never assert these are true rooms.

    The selected height slice is rasterized as occupied wall evidence. Small
    gaps are closed, and enclosed empty regions are measured on the grid.
    Doorways, missing walls, furniture, and scan drift can change the result.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError("points must have shape N x 3")
    if not np.isfinite(cloud).all():
        raise ValueError("points must contain only finite coordinates")
    if grid_size_m <= 0 or slice_min_height_m < 0 or slice_max_height_m <= slice_min_height_m:
        raise ValueError("grid and slice dimensions must be positive and ordered")
    if not np.isfinite(grid_rotation_deg):
        raise ValueError("grid rotation must be finite")
    if gap_closing_cells < 1 or min_area_m2 <= 0 or max_area_m2 <= min_area_m2:
        raise ValueError("closing size and area limits must be positive and ordered")
    if min_dimension_m <= 0:
        raise ValueError("minimum room dimension must be positive")
    if len(cloud) == 0:
        return []

    floor_y = estimate_floor_height(cloud)
    in_slice = cloud[
        (cloud[:, 1] >= floor_y + slice_min_height_m)
        & (cloud[:, 1] <= floor_y + slice_max_height_m)
    ]
    if len(in_slice) < 2:
        return []

    angle = np.deg2rad(grid_rotation_deg)
    basis = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    local_xz = in_slice[:, (0, 2)] @ basis
    cells = np.unique(np.floor(local_xz / grid_size_m).astype(np.int64), axis=0)
    low = cells.min(axis=0)
    high = cells.max(axis=0)
    occupied = np.zeros(high - low + 1, dtype=bool)
    occupied[cells[:, 0] - low[0], cells[:, 1] - low[1]] = True
    padding = gap_closing_cells + 2
    occupied = np.pad(occupied, padding)

    closed = ndimage.binary_closing(
        occupied,
        structure=np.ones((3, 3), dtype=bool),
        iterations=gap_closing_cells,
    )
    enclosed = ndimage.binary_fill_holes(closed) & ~closed
    labels, count = ndimage.label(enclosed)

    candidates = []
    for label_id in range(1, count + 1):
        indices = np.argwhere(labels == label_id)
        area = len(indices) * grid_size_m**2
        if not min_area_m2 <= area <= max_area_m2:
            continue
        lower = indices.min(axis=0)
        upper = indices.max(axis=0) + 1
        dimensions = (upper - lower) * grid_size_m
        if float(dimensions.min()) < min_dimension_m:
            continue
        offset = low - padding
        local_center = (indices.mean(axis=0) + offset + 0.5) * grid_size_m
        center = local_center @ basis.T
        boundary = _outer_boundary(labels[lower[0]:upper[0], lower[1]:upper[1]] == label_id)
        boundary_local = np.asarray([
            [(point[0] + lower[0] + offset[0]) * grid_size_m,
             (point[1] + lower[1] + offset[1]) * grid_size_m]
            for point in boundary
        ])
        boundary_world = boundary_local @ basis.T
        boundary_xz = [[round(float(value), 3) for value in point] for point in boundary_world]
        world_minimum = boundary_world.min(axis=0)
        world_maximum = boundary_world.max(axis=0)
        candidates.append({
            "id": f"room_candidate_{len(candidates) + 1:02d}",
            "estimated_area_m2": round(float(area), 2),
            "bounds_xz_m": [
                [round(float(world_minimum[0]), 3), round(float(world_minimum[1]), 3)],
                [round(float(world_maximum[0]), 3), round(float(world_maximum[1]), 3)],
            ],
            "center_xz_m": [round(float(v), 3) for v in center],
            "boundary_xz_m": boundary_xz,
            "status": "candidate_unverified",
            "reason": (
                "Enclosed grid region after closing small gaps in the height slice; "
                f"analysis grid rotated {grid_rotation_deg:.1f}° to dominant wall direction; "
                "review for doors, missing walls, and scan drift."
            ),
        })
    return candidates
