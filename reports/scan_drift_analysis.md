# Repeat-pass consistency check

This is a consistency check between the two passes in
`single_scan_with_ceiling.zip`. It is not measured drift or ground-truth
accuracy: the capture does not verify that the operator followed the same path
or returned to the exact starting point.

## Findings

The camera-pitch split is at row **5,200** (**112.87 s**). Pass A is 52.84 m
long with a 0.36 m recorded endpoint gap; pass B is 46.92 m long with a 0.37 m
gap. The recorded paths overlap, but nearest-path distance is asymmetric:
the 95th percentile is 0.586 m from A to B and 0.271 m from B to A.

The wall-height point sets already overlap closely before alignment. The 95th
percentile nearest-wall distance is 0.167 m from B to A and 0.267 m from A to
B. A single rigid fit estimates only **−0.008°** rotation and about **2 mm**
translation. After that fit, the corresponding distances are 0.166 m and
0.266 m. This is a negligible change, so this run provides no evidence that a
global rigid adjustment improves the fused wall cloud. The reconstruction
keeps the recorded poses unchanged.

The difference between path overlap and wall-surface overlap, and the
directional asymmetry, mean these values should be read as repeatability
diagnostics only. A smoothly varying correction is not justified by this
global fit; it needs independent local correspondences and could otherwise
warp the room.

## Reproduce

```sh
PYTHONPATH=src python scripts/analyze_scan_drift.py \
  data/raw/single_scan_with_ceiling.zip \
  --frame-step 10 --voxel-size 0.04 \
  --output-dir runs/drift_analysis
```

The command reads the ZIP without extracting it and writes a JSON report and
plot under `runs/drift_analysis/`.
