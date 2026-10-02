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

The project targets Python 3.11 and 3.12. Runtime dependencies are intentionally
empty at this stage. No data or packages are downloaded or installed by this
scaffold.
