# Room Reconstruction

A Python project for room reconstruction workflows.

## Project layout

- `data/raw/` — local raw captures (ignored by Git).
- `data/ground_truth/` — ground-truth data.
- `src/room_reconstruction/` — installable Python package source.
- `tests/` — automated tests.
- `reports/` — generated reports and analysis outputs.
- `scripts/` — command-line and utility scripts.

## Inspecting raw archives

To summarize the three expected ZIP archives without extracting them, run:

```sh
python scripts/inspect_archives.py
```

You can also name specific archives and choose how many member paths to sample:

```sh
python scripts/inspect_archives.py data/raw/single_room.zip --sample-size 10
```

The script reads ZIP metadata only, flags absolute or parent-traversal member
paths, and exits nonzero if any requested archive is missing or malformed.

## Loading one depth frame

The point-cloud code uses NumPy, Pillow, and SciPy; CSV parsing uses Python's
standard library. Matplotlib renders the parameter-comparison plot.

To combine every tenth odometry frame and keep one point per 5 cm voxel:

```sh
python -c 'from room_reconstruction.pointcloud import load_capture_point_cloud; p = load_capture_point_cloud("data/processed/single_room/c00a170fe1", frame_step=10, voxel_size=0.05); print(p.shape, p.min(axis=0), p.max(axis=0))'
```

This call returns an `N x 3` NumPy array in world coordinates; it does not write
an output file.

To render a top-down comparison of several frame-step and voxel-size settings:

```sh
python scripts/compare_pointcloud_parameters.py
```

The comparison is saved to `reports/pointcloud_parameter_comparison.png`.

To process the full `single_room` capture using every tenth frame and 2 cm
voxels, save the compressed point cloud, and render its top-down view:

```sh
python scripts/process_pointcloud.py
```

The command writes `reports/single_room_step10_voxel2cm.npz` and
`reports/single_room_step10_voxel2cm.png`.

The same command can read a ZIP capture directly without extracting it. For
example:

```sh
python scripts/process_pointcloud.py data/raw/single_scan_with_ceiling.zip
```

To inspect odometry continuity for the three capture archives without
extracting any files, run:

```sh
python3 scripts/inspect_trajectory.py
```

The report shows frame count, duration, recorded position bounds, per-frame
translation and rotation, and unusually large pose steps. These checks can
flag discontinuities, but they do not measure accumulated tracking drift or
the accuracy of the reconstructed room.

For the two-pass ceiling-inclusive capture, compare the pitch-based pass split,
recorded route closure, and wall-slice overlap before and after a single rigid
alignment step:

```sh
PYTHONPATH=src python scripts/analyze_scan_drift.py
```

The ZIP is read directly without extraction. The command writes
`runs/drift_analysis/drift_analysis.json` and `.png`. The rigid transform is a
diagnostic ablation only; it is not applied to the reconstruction because a
single transform cannot correct drift that changes along the route. All
reported overlaps are repeatability checks, not ground-truth accuracy.

To compare three strengths of a statistical outlier filter on the
ceiling-inclusive point cloud, run:

```sh
PYTHONPATH=src python scripts/compare_pointcloud_filter.py
```

The script writes a four-panel image to
`reports/single_scan_with_ceiling_filter_strengths.png`. Each variant uses
the same neighbor distances and changes only the outlier threshold. The filter
is an experiment: it may remove real isolated surfaces along with noise, so
the unfiltered cloud remains unchanged. It reads the ZIP directly; use
`--input path/to/cloud.npz` to compare a saved point cloud instead.

To see where the ceiling-inclusive scan has point support at different heights,
run:

```sh
PYTHONPATH=src python scripts/inspect_scan_coverage.py
```

This writes `reports/single_scan_with_ceiling_coverage.png`, with one map for
the floor, lower wall, room-detection, and upper wall/ceiling bands. Empty
cells indicate little or no point support at that height; they are a capture
coverage diagnostic, not proof that a surface is physically absent.

To check whether the room and wall candidates change when we use other wall
heights, run:

```sh
PYTHONPATH=src python scripts/compare_vertical_slices.py
```

This compares lower, current, upper, and combined height bands in
`reports/single_scan_with_ceiling_vertical_slices.png`. It is an experiment
for choosing where to look for geometry; candidate changes do not establish
which band is more accurate.

To highlight wall segments that line up across multiple height bands, run:

```sh
PYTHONPATH=src python scripts/compare_wall_consensus.py
```

The unverified agreements are saved to
`reports/single_scan_with_ceiling_wall_consensus.json` and previewed in the
matching PNG. This uses geometric tolerances for angle, offset, and overlap;
agreement only shows repeatability across slices.

## Running the reconstruction app

The first runnable milestone accepts one LiDAR capture directory or ZIP. It
applies the confidence mask, transforms sampled depth pixels into world
coordinates, and writes a point cloud, previews, and a versioned JSON result.
It reports unverified room and wall candidates, not confirmed room geometry or
measurements. The output contract is documented in `schemas/plan.schema.json`.

```sh
PYTHONPATH=src python run.py data/raw/single_scan_with_ceiling.zip \
  --tier lidar --frame-step 10 --voxel-size 0.02 --output-dir runs/ceiling
```

The output directory contains `point_cloud.npz`, `plan.png`, and `plan.json`.
Photo and video capture tiers are not wired up yet; the command reports that
clearly instead of returning fabricated geometry.

The command also records and renders tentative wall traces in
`wall_candidates.json` and `wall_candidates.png`. It estimates floor height,
uses a horizontal slice about 0.85–1.55 m above that estimate, and fits line
segments to the slice. Each wall entry now lists which lower, current, and
upper height bands have the strongest matching evidence on an overlapping
subsection; `cross_height_consensus_ids` links to the exact shared segments.
This count is a consistency hint, not a confidence or accuracy score. Shared
line segments are written to
`cross_height_wall_candidates.json` and rendered in
`cross_height_wall_candidates.png`. All remain unverified candidates.
It also lists enclosed-region candidates in `room_candidates.json` and
`room_candidates.png`. Dashed outlines trace the occupied grid boundary; they
are candidates, not verified room geometry or survey-grade floor areas.
For the room raster, the grid is rotated to the dominant orientation of
cross-height wall traces. This reduces axis-aligned raster artifacts while
leaving the traced polygon free-form; it does not snap walls to right angles
or force the schematic room shape onto the scan.
The runner also proposes four-sided outlines in `wall_room_candidates.json` by
joining near-parallel opposing wall segments that are approximately
perpendicular to a second pair. An outline is only emitted when the corners
are near right angles and at least 70% of every side is directly supported by
the scan. Each outline records its source segments and side support fractions.
An empty list means the scan did not support a closed outline at that
threshold; it does not mean the physical room is absent.
Possible openings between aligned wall candidates are written to
`opening_candidates.json`; each proposed room connection requires room
candidates on opposite sides of that gap. Short aligned interruptions are also
listed in `wall_gap_candidates.json`, even when the other side has no room
candidate. These gaps are not classified as doorways automatically; the review
page shows them in a separate Door / gaps tab.
`plan.json` keeps the verified room list empty, and openings, adjacencies,
dimensions, and floor areas still need validation.
Current floor-only and floor-and-ceiling candidate outputs are compared in
[`reports/scan_candidate_comparison.md`](reports/scan_candidate_comparison.md).
The schematic room reference and current scan comparison are summarized in
[`reports/ground_truth_comparison.md`](reports/ground_truth_comparison.md).
The left and right wall lengths were remeasured as distinct sides; the room
outline and corner geometry still need to be recorded before comparing area.

To review the existing ceiling-inclusive reports locally in a browser, run:

```sh
python scripts/review_scan.py reports --report-prefix single_scan_with_ceiling
```

For a new result folder created by the LiDAR command above, point the review
page at that folder instead:

```sh
python scripts/review_scan.py runs/ceiling
```

Open the printed localhost address. The page shows the point-cloud preview and
raster room, wall-supported outline, wall, and cross-height wall candidates. Candidate review marks stay in
your browser; **Export review JSON** downloads them as `scan_review.json`.
The review server reads result files and does not modify them. Stop it with
Ctrl+C. The earlier report format does not include a point count; newer result
folders include it.

Set up the project and run the focused tests with:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The project targets Python 3.11 and 3.12. Runtime dependencies are declared in
`pyproject.toml`. The setup commands install the project and its dependencies;
they do not download or modify dataset files.
