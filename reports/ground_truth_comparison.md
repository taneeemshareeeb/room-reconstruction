# Room layout comparison

The current measurements are read from the fresh hand-drawn reference photo
[`data/raw/IMG_7647.HEIC`](../data/raw/IMG_7647.HEIC). The sketch is not treated
as the exact room outline. Values are user measurements transcribed from the
photo; small centimeter labels around the steps could not be assigned to
specific edges reliably.

## Current measurements

| Measurement | Value | Reading |
| --- | ---: | --- |
| Top width | 3.14 m | Written across the top wall |
| Left wall | 3.72 m | Written alongside left side |
| Right wall | 3.77 m | Written alongside right side |
| Bottom wall before doorway | 2.65 m | Partial wall segment |
| Doorway | 0.93 m | Bottom wall, near right corner |
| Ceiling height | 2.71 m | Height written on sketch |

The prior sketch values are preserved as superseded history in
[`data/ground_truth/room_layout.json`](../data/ground_truth/room_layout.json).
The bottom wall measurement is a partial segment, not a full room width.

## Comparison with the ceiling-inclusive scan

Candidate 04 has rough dominant-axis spans of **3.20 × 3.44 m**; the closest
span differs from the measured 3.14 m top width by **0.06 m**. Candidate 03
has a rough 3.60 m span, **0.12 m** from the measured 3.72 m left side and
**0.17 m** from the 3.77 m right side. These polygon extents are not fitted
wall lengths, and the scan is not spatially registered to the sketch.

Candidate 04 also has nearby `wall_candidate_07`: it is **3.699 m** long,
15.6 cm from the region boundary, and supported in all three height bands.
The comparison now assigns distinct scan traces to the two side measurements
within each region. Candidate 04 has one compatible match: `wall_candidate_07`
is 2.1 cm from the 3.72 m left-side measurement. Its separate trace for the
3.77 m right side is 3.384 m, 38.6 cm short. Candidate 04 therefore has one
supported side-length match, not two. These assignments use length only and do
not establish which scan trace corresponds to which physical wall.

In Candidate 04 the upward-pass heights are **3.075 m** (28,346 points) and
**2.475 m** (2,356 points). Relative to the 2.71 m measurement, these differ
by +0.365 m and −0.235 m respectively. The denser peak is higher; the scan
does not establish which band represents this room's ceiling.

The closest wall gap remains **0.819 m**, **0.111 m (11.9%)** narrower than
the newly measured 0.93 m doorway. It is not attached to a room boundary, so
the width comparison cannot identify it as the doorway. That gap lies 6.50 m
from Candidate 04; Candidate 04's nearest gap is 1.06 m away and is 0.46 m
wide.

The run still produces four raster room candidates and no fully supported
wall outline. Candidate 04 is the strongest measurement lead, with one of two
side lengths supported by a distinct nearby trace. The doorway gap is still
not spatially associated with that candidate. Room identity and the complete
boundary remain unresolved.

Reproduce the comparison with:

```sh
PYTHONPATH=src python run.py data/raw/single_scan_with_ceiling.zip \
  --tier lidar --frame-step 10 --voxel-size 0.02 \
  --output-dir runs/single_scan_with_ceiling \
  --ground-truth data/ground_truth/room_layout.json
```
