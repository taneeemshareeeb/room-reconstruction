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
