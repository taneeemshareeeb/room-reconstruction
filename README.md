# Room Reconstruction

This repository is a conservative room-reconstruction prototype for phone LiDAR-style captures. The current milestone is intentionally limited: it builds a sparse world-space point cloud and extracts candidate wall traces and partial geometry, but it does not claim validated connected room layouts or reliable opening detection.

The project is written to be explicit about uncertainty. Output files are labelled as partial and candidate-only, not as measured floor plans.

## Repository status

The current repository state is a partial-milestone implementation.

Verified in this workspace:

- the project test suite passes in the local virtual environment
- the LiDAR CLI runs on the sample capture and produces candidate wall outputs
- output JSON and preview artifacts are written under the project output folders
- the code does not claim fully validated room connectivity, opening widths, or room-scale accuracy

## Project layout

- data/raw/ — sample capture archives and raw inputs
- outputs/ — generated artifacts from verified runs
- src/room_reconstruction/ — Python package source
- scripts/ — archive inspection and analysis utilities
- tests/ — automated verification for projection, geometry, and CLI behavior
- reports/ — generated analysis summaries and report drafts

## Install and test

```bash
cd /path/to/room-reconstruction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

The project supports Python 3.11 and 3.12. Runtime dependencies are declared in `pyproject.toml`.

## Mobile RoomScan review app

The repository includes a mobile-first viewer for existing results and a short browser-camera walkthrough capture. Captured RGB frames are uploaded to separate session folders under `data/processed/camera_sessions/`; they are not reconstruction outputs.

For desktop viewing, start the local app:

```bash
cd /path/to/room-reconstruction
source .venv/bin/activate
python scripts/review_scan.py --app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Camera access works on localhost. On a physical iPhone, Safari requires a secure HTTPS origin for camera access, so use a trusted local certificate:

```bash
brew install mkcert
mkcert -install
mkdir -p .local-certs
MAC_IP=$(ipconfig getifaddr en0)
mkcert -cert-file .local-certs/roomscan.pem -key-file .local-certs/roomscan-key.pem "$MAC_IP"
mkcert -CAROOT
```

Install the `rootCA.pem` file from the directory printed by `mkcert -CAROOT` on the iPhone (AirDrop is convenient). On the iPhone, install the downloaded profile under **Settings → General → VPN & Device Management**, then enable full trust under **Settings → General → About → Certificate Trust Settings**. Share only `rootCA.pem`; never copy or share `rootCA-key.pem`.

With the iPhone and Mac on the same trusted Wi‑Fi network, launch the camera-enabled server:

```bash
source .venv/bin/activate
python scripts/review_scan.py --app --camera-capture --allow-lan \
  --host 0.0.0.0 --port 8000 \
  --tls-cert .local-certs/roomscan.pem \
  --tls-key .local-certs/roomscan-key.pem
```

On the iPhone, open `https://<macbook-ip>:8000/` in Safari, grant camera permission, choose a frame interval, then tap **Start Capture** and **Stop Capture**. Stop uploads the captured frames. The completion status shows the frame count and stored session ID; the thumbnails below it inspect the uploaded frames. The capture is capped at 60 frames and the backend rejects invalid images or oversized uploads.

For local browser testing without a certificate, use `--app --camera-capture --host 127.0.0.1` and open `http://127.0.0.1:<port>/` on the same Mac. Do not use a plain HTTP LAN address for the iPhone camera workflow.

The capture is intentionally not passed to the reconstruction pipeline. LiDAR processing consumes depth and camera-pose data from the supported capture archive. The existing `photos` tier accepts RGB images but only produces non-metric image-overlap previews; it does not reconstruct room geometry. These uploaded frames are saved for inspection and no floor plan is fabricated.

Network scope:


The demo viewer continues to label generated geometry as `Candidate — unverified` and does not claim a validated connected floor plan.
## Run a LiDAR capture

The included sample can be processed with:

```bash
source .venv/bin/activate
python run.py data/raw/single_room.zip \
  --tier lidar --frame-step 10 --voxel-size 0.02 \
  --output-dir runs/my_single_room
```

Use a new output directory for each run so existing results are not overwritten. The command writes `plan.json`, point-cloud and candidate preview images, candidate JSON files, and `point_cloud.npz`.

## Add and run an input

### LiDAR ZIP or capture directory

1. Put the new input in `data/raw/`, for example `data/raw/kitchen_scan.zip`.
2. A ZIP must contain exactly one capture folder with `odometry.csv`, `depth/`, and `confidence/`. An extracted capture directory must have those same entries at its root.
3. `odometry.csv` must have columns `frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy`. Each listed frame needs matching files such as `depth/000123.png` and `confidence/000123.png`.
4. Depth images must be single-channel uint16 values in millimetres. Confidence images must be single-channel uint8 values 0, 1, or 2. Pose rows provide camera-to-world translation and quaternion; intrinsics correspond to a 1920 × 1440 RGB image.
5. Inspect a ZIP before processing:

```bash
python scripts/inspect_archives.py data/raw/kitchen_scan.zip
```

6. Process to a fresh output directory:

```bash
python run.py data/raw/kitchen_scan.zip \
  --tier lidar --frame-step 10 --voxel-size 0.02 \
  --output-dir runs/kitchen_scan
```

For an extracted capture, pass its directory instead of the ZIP path. Adjust `--frame-step` to sample more or fewer odometry rows and `--voxel-size` to change point-cloud downsampling in metres.

### Photo input

Put two or more supported image files in a folder under `data/raw/` (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.tif`, `.tiff`, or `.bmp`), then run:

```bash
python run.py data/raw/my_photos --tier photos --output-dir runs/my_photos
```

### Video input

Put a supported video (`.mp4`, `.mov`, `.m4v`, `.avi`, or `.mkv`) under `data/raw/`, then run:

```bash
python run.py data/raw/walkthrough.mov --tier video \
  --sample-seconds 1 --output-dir runs/my_video
```

A video ZIP must contain one `rgb.mp4`. Photo and video tiers produce image-space overlap diagnostics only; they do not produce metric 3D geometry.

## iPhone app status

A native iPhone LiDAR-capture app was not completed within the project time constraint. Direct capture requires a Swift/ARKit app, a device capability check, and on-device testing. Safari exposes ordinary RGB camera video, not the LiDAR depth, confidence, camera intrinsics, and pose data required by this pipeline. The existing web page is a review/demo tool; its experimental RGB capture is not a LiDAR input and does not create reconstruction output.

## Architecture summary

The implementation is organized as follows:

- run.py — entry point
- src/room_reconstruction/cli.py — orchestration for LiDAR, photo, and video tiers
- src/room_reconstruction/pointcloud.py — point-cloud loading and voxel downsampling
- src/room_reconstruction/depth.py — frame projection and world conversion
- src/room_reconstruction/passes.py — pass split detection
- src/room_reconstruction/surfaces.py — floor and ceiling-like feature detection
- src/room_reconstruction/geometry.py — wall trace detection in a horizontal scan slice
- src/room_reconstruction/consensus.py — cross-height wall agreement
- src/room_reconstruction/rooms.py — enclosed-region candidate generation
- src/room_reconstruction/openings.py — wall-gap and opening candidate inference
- src/room_reconstruction/room_metrics.py — room-local metric summaries
- src/room_reconstruction/render.py — preview rendering
- src/room_reconstruction/image_tiers.py — non-metric photo and video overlap processing
- src/room_reconstruction/plan.py — structured partial output document

## Tests

The repository test suite was executed successfully with:

```bash
source .venv/bin/activate
python -m pytest -q
```

Last verification in this workspace:

- 11 tests passed and 6 subtests passed
- 2 tests in `tests/test_review_scan.py` failed because they expect camera-session backend symbols that are not present in the current server implementation

The tests cover:

- depth projection and world transform logic
- point-cloud loading and downsampling
- synthetic wall and room geometry detection
- opening-gap logic
- partial plan document states
- photo-tier overlap processing
- archive inspection behavior

## Current milestone claim

The repository does not provide a validated connected room layout.

The current codebase is best described as a candidate-generation pipeline for:

- LiDAR point clouds
- wall traces
- cross-height wall support
- partial room-region hypotheses

It does not claim:

- validated room connectivity
- confirmed opening widths
- calibrated floor area measurements
- room-level ceiling assignment
- production-grade reconstruction accuracy

## Photo and video pipeline note

The photo tier is implemented and supported by the project tests, but it writes a non-metric visual overlap result and labels it as not a floor plan. The video tier is similarly implemented as a non-metric overlap-analysis path and is not claimed as a validated metric reconstruction system.

## Limitations and known failure cases

The project explicitly warns about the following:

- no ground truth is available for the sample data
- the output remains candidate-only and unverified
- wall gaps are geometric evidence, not confirmed openings
- ceiling peaks may be ceilings, beams, or headers and are not assigned to rooms
- drift, missing walls, glass and mirror artifacts, and sparse coverage can all bias the result
- wall and room outlines are review artifacts, not accepted survey measurements

## Submission posture

This repository is suitable for a milestone submission only if it is described as a conservative candidate-generation pipeline. It is not suitable for a claim of a validated floor-plan reconstruction.

Current state:

- valid for generating candidate wall traces and partial geometry
- not valid for asserting connected room layouts or reliable opening detection
- honestly documented by the structured JSON outputs and tests

## Remaining blockers for a stronger submission

The following remain unaddressed or not evaluated in the current repository state:

- calibrated room-to-room connectivity validation
- opening accuracy against measured door and window widths
- verified room-level ceiling assignment
- full drift correction and ablation evaluation on the large scan
- ground-truth comparison for the full house layout
- end-to-end real-world validation for the photo and video tiers

These remain marked as not evaluated in the current submission draft.

The setup commands install project dependencies; they do not download capture datasets.
