# Room layout comparison

The reference values come from the user-provided sketch and remeasurements in
[`data/ground_truth/room_layout.json`](../data/ground_truth/room_layout.json).
The current comparison was generated from `single_scan_with_ceiling.zip` using
every tenth frame and 2 cm voxels. The sketch is a location reference, not a
registered floor plan, and the room shape is not forced to be rectangular.

## Recorded measurements

| Measurement | Value | Status |
| --- | ---: | --- |
| Top width | 3.36 m | Recorded from sketch |
| Bottom width | 3.32 m | Recorded from sketch |
| Left wall | 3.63 m | Remeasured |
| Right wall | 4.15 m | Remeasured; kept separate |
| Ceiling height | 2.63 m | Confirmed by user |
| Doorway width | 0.85 m | Confirmed by user |

The left and right wall lengths differ by 0.52 m. They are not averaged. The
full perimeter and corner locations have not been measured, so these values do
not define a reference area or a complete reference polygon.

## Scan comparison

The scan still yields four raster room candidates and no closed wall-supported
room outline. Candidate 04 is the closest match to the two similar width
measurements: its rough dominant-axis spans are **3.20 × 3.44 m**, with a mean
absolute difference of about **0.10 m**. Its raster footprint is **8.08 m²**.
These are occupied-grid region estimates, not fitted wall lengths or a
confirmed match to the measured room.

The per-room floor estimates range from **−1.495 m to −1.482 m** in world y,
with 18,114–41,079 supporting points per candidate. The consistent floor
levels support the gravity-aligned floor estimate, but do not validate any
room boundary.

The scan has different upper-surface peaks in different regions. Candidate 04
has peaks at **2.475 m** (2,356 points) and **3.075 m** (28,346 points) above
its local floor. Against the measured **2.63 m** ceiling, the lower peak is
closer by 0.155 m, while the stronger peak is 0.445 m higher. This is
ambiguous evidence: the weaker peak could be a beam or header, and the larger
peak could belong to another height level or adjacent space. No ceiling is
confirmed.

The closest scan wall gap is **0.819 m**, **0.031 m (3.6%)** narrower than the
measured 0.85 m doorway. That gap is not spatially associated with one of the
four room polygons, so width similarity alone cannot identify it as this
doorway.

The measured left wall is numerically closest to a **3.60 m** rough span in
Candidate 03 (0.03 m difference). The measured right wall is also closest to
Candidate 03's 3.60 m rough span (0.55 m difference). The width-matched
Candidate 04 is different, and its spans do not explain the longer measured
side. No single candidate currently accounts for the full set of dimensions.

## Current result and limits

- Each raster region now has a rough footprint, nearby wall and wall-gap
  references, floor support, and upward-pass height peaks.
- Measured widths, wall lengths, ceiling height, and doorway width are compared
  numerically in
  [`runs/single_scan_with_ceiling/ground_truth_comparison.json`](../runs/single_scan_with_ceiling/ground_truth_comparison.json).
- The best numerical width match is still only a hypothesis. Scan and sketch
  coordinates are not registered, and rough extents are not wall dimensions.
- Missing room closure and the conflicting upper-height peaks mean the scan
  does not yet provide a complete, verified room plan. Review the per-room
  candidates before using dimensions for decisions.

The remaining evidence needed for a verified layout is a measured room
perimeter with corner positions (or a scan-to-sketch registration), plus a
visual check of which upper surface is the actual ceiling and which wall gap is
the doorway.
