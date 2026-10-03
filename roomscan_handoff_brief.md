# Room-scan take-home: handoff brief

Paste this whole file into the coding assistant. Everything marked **VERIFIED** was checked on the actual sample data. Everything marked **UNVERIFIED** is a hypothesis; test it before relying on it.

## 1. The task (short)

Turn phone captures of a house into a dimensioned, stitched floor plan with an honest confidence interval on every number. Output per capture: JSON (walls, wall lengths, ceiling height, floor area, openings, adjacency) plus a rendered plan image, from ONE command. Three input tiers are in the spec (LiDAR, video, photos); the company's reply says the Drive folder is the sample data to run on. Deadline: Oct 4, 10:00 AM IST.

Key grading ideas from the spec:
- Honest uncertainty beats confident wrong answers (confident garbage on thin input caps the score).
- Opening (door/window) widths target 2 cm on 85% of openings; a missed AND a phantom opening both count as misses.
- Ceiling height target 1.5 cm; repeat captures should agree within 1 cm.
- Drift must be corrected and shown with an on/off ablation ("poses used as-is" fails).
- Code must run cold on a new capture, one command, README that works in under 15 minutes on a clean machine.
- Commit history is read, so commit continuously with meaningful messages.

## 2. The sample data (VERIFIED)

Three zips, each one folder in the format of the iPhone LiDAR logging app Stray Scanner:

| Zip | Frames | Duration | What it is |
|---|---|---|---|
| single_room.zip | 1,715 | 37 s | A corner section of a home, a few rooms |
| single_scan_floor_only.zip | 5,251 | 115 s | Whole apartment/house, one continuous walk, ceiling NOT scanned |
| single_scan_with_ceiling.zip | 9,745 | 215 s | Whole house walked TWICE: pass A (walls/floor) then pass B (ceiling) |

Per folder:
- `rgb.mp4`: 1920x1440, 60 fps
- `depth/000000.png ...`: 256x192, uint16, millimetres (divide by 1000 for metres)
- `confidence/000000.png ...`: 256x192, values 0/1/2 (2 = most trusted)
- `odometry.csv`: header `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y` (one row per frame; note the spaces after commas, use `skipinitialspace=True`; `frame` is zero-padded like `000000`)
- `camera_matrix.csv`: 3x3 intrinsics for the 1920x1440 image
- `imu.csv`: `timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z`

## 3. Verified findings

1. **Intrinsics must be rescaled.** fx, fy, cx, cy are for the 1920-wide RGB image; depth is 256 wide, so multiply by 256/1920 (= 1/7.5).
2. **Camera axis convention: use plain OpenCV (x right, y down, z forward), with NO flip.** I first assumed the ARKit convention (flip y and z); that produced a smeared cloud. I tested six conventions and scored cloud sharpness: no-flip with `world = cam @ R.T + t` was about 2.5x sharper than the next best. Pose rotation is `scipy Rotation.from_quat([qx, qy, qz, qw])`.
3. **World +y is up (gravity-aligned).** The floor shows as a very sharp peak in the y histogram (about y = -1.50 m in single_room and with_ceiling, -1.41 m in floor_only). Phone height is about 1.5 m above the floor at start.
4. **Ceiling not observed in single_room or floor_only** (highest points only slightly above phone height). The pipeline must return "ceiling not observed" with a wide or null interval, not a guess.
5. **with_ceiling** has ceiling-like peaks in the y histogram near +0.95 m and +1.6 m. If the upper one is the ceiling that is about 3.1 m above the floor; the lower (about 2.45 m above the floor) could be a lowered ceiling or door headers. UNVERIFIED which; check per room using pass B only.
6. **with_ceiling pass split (VERIFIED from camera pitch):** the camera view direction y-component is about -0.45 (looking down) for t < ~112 s and +0.2 to +0.5 (looking up) after. The walking route (x,z) in both halves is nearly identical. Split at about 112 s, or detect it from pitch.
7. **Drift evidence:** start-to-end gaps of the phone path: pass A 0.21 m (52 m path), pass B 0.44 m (46.8 m path), floor_only 0.17 m (53.8 m path). These measure drift only if the person ended exactly where they started (UNVERIFIED). The two passes' position traces also differ by up to roughly 0.5 m at some points (read from a plot, not yet computed). Some walls look doubled in the top-down slice of the large scans, consistent with drift.
8. **Possible artifact:** single_room shows a long thin diagonal line in the top-down slice that does not look like a wall (possibly glass/mirror depth error). Needs filtering without deleting real walls.
9. **UNVERIFIED:** floor_only might be the same house as pass A of with_ceiling (similar duration 115 s vs 112 s, path 53.8 m vs 52.0 m, PCA extents 7.7x8.7 vs 8.8x8.6, so not a perfect match). Yaw is arbitrary per session, so test by registering top-down slices (rotate/translate search or ICP). If it is the same house, it gives a cross-session repeatability check.

## 4. Verified loader (works; runs in about 2 s per capture at step=5)

```python
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

def load_capture(root, step=5, min_conf=2, max_depth_m=5.0, frame_range=None):
    root = Path(root)
    odo = pd.read_csv(root / "odometry.csv", skipinitialspace=True)
    idx = range(0, len(odo), step) if frame_range is None else range(frame_range[0], frame_range[1], step)
    pts_all = []
    for i in idx:
        row = odo.iloc[i]
        name = f"{int(row['frame']):06d}.png"
        depth = np.array(Image.open(root / "depth" / name)).astype(np.float32) / 1000.0
        conf = np.array(Image.open(root / "confidence" / name))
        h, w = depth.shape
        s = w / 1920.0                                   # intrinsics are for the 1920-wide RGB
        fx, fy, cx, cy = row.fx * s, row.fy * s, row.cx * s, row.cy * s
        v, u = np.mgrid[0:h, 0:w]
        ok = (conf >= min_conf) & (depth > 0.1) & (depth < max_depth_m)
        z = depth[ok]
        x = (u[ok] - cx) * z / fx
        y = (v[ok] - cy) * z / fy
        cam = np.stack([x, y, z], 1)                     # OpenCV convention, NO axis flip
        rot = R.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
        pts_all.append(cam @ rot.T + np.array([row.x, row.y, row.z]))
    return np.concatenate(pts_all)

def voxel_downsample(pts, voxel=0.02):
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx]
```

Sanity check to keep in the repo as a test: after loading, the y histogram must show one very sharp floor peak. If it does not, the axis convention is wrong.

## 5. Implementation plan (priority order)

1. **Pass split** for with_ceiling by camera pitch (view direction = `R @ [0,0,1]`, y-component smoothed over about 1 s; looking up when > 0.1).
2. **Floor and ceiling:** floor = sharpest low peak in y histogram (use pass A). Ceiling = peak from pass B points only. Report per-room ceiling heights if distinct levels exist. Ceiling height = ceiling_y - floor_y, with the interval from the peak width plus the vertical drift between passes.
3. **Wall extraction:** slice points between about floor + 0.3 m and floor + 1.8 m, project to top-down (x,z), find dominant wall directions (histogram of local normals or Hough), snap to Manhattan axes where the evidence supports it, fit and clean wall lines, build room polygons, wall lengths, floor area. Keep an escape hatch for non-orthogonal rooms.
4. **Room segmentation and adjacency** for the whole-house scans (free space in the top-down view, split at doorways).
5. **Openings:** gaps or depth discontinuities in wall planes between floor and about 2.1 m; classify door vs window by sill height. Time-box this; report recall and phantom rate honestly.
6. **Drift handling + ablation:** use the two passes of with_ceiling. Estimate the offset between the passes along the route (match pass A and pass B trajectories, then align the clouds, e.g. ICP on the top-down wall slice), apply a smooth correction along the path, and show the stitched footprint with the correction off and on. Report the gap between passes before and after.
7. **Uncertainty:** bootstrap over random frame subsets plus a drift term. State plainly that intervals are NOT calibrated against ground truth (see section 6).
8. **Output:** one command (`python run.py <capture_dir> --tier lidar`) writing `plan.json` and `plan.png`. JSON must include a confidence interval on every number.
9. **Photo and video tiers:** if time is short, implement clearly labelled stubs that return very wide intervals and a reason string. Do not claim accuracy that has not been measured.
10. **Report (max 6 pages) and README:** architecture, drift handling with the ablation, error budget, known failure modes (glass/mirrors, low light, ceiling not observed), what was and was not validated.

## 6. Honesty constraints (important)

- **No ground truth exists for the sample data**; the company said we must take measurements ourselves. We also have no LiDAR iPhone, so we cannot capture our own LiDAR data.
- Therefore the only accuracy-type evidence available is **consistency**: pass A vs pass B, floor_only vs with_ceiling (if same house), and bootstrap spread. Label these as consistency numbers, never as accuracy.
- Do not say "within X cm of true size" anywhere unless it was measured against a tape or laser.
- Where the ceiling is not observed, output null plus a reason, not a number.

## 7. Open questions

- Is the sample data the only input expected, and are the photo and video tiers required in this round? (Email sent; the reply only confirmed the data is sample data.)
- Are floor_only and with_ceiling the same house?
- Which ceiling peak in with_ceiling is the real ceiling, per room?
