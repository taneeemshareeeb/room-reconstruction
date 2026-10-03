# Room layout comparison

This comparison uses the manually measured layout in
[`data/ground_truth/room_layout.json`](../data/ground_truth/room_layout.json)
and the current ceiling-inclusive run produced with frame step 10 and 2 cm
voxels. The sketch is schematic; the room may not have parallel sides or a
rectangular outline.

## Measurements

| Measurement | Value | Reading |
| --- | ---: | --- |
| Top width | 3.36 m | Recorded in sketch |
| Bottom width | 3.32 m | Recorded in sketch |
| Left length | 3.63 m | Remeasured; retain as its own side |
| Right length | 4.15 m | Remeasured; retain as its own side |
| Ceiling height | 2.63 m | Confirmed by user |
| Doorway width | 0.85 m | Confirmed by user; bottom wall near right corner |

The two widths agree within 0.04 m. The remeasured side lengths differ by
0.52 m; do not average them or use them as opposite sides of a rectangle.
These lengths alone do not determine the room area because the corner geometry
and full perimeter are not recorded.

## Current scan comparison

The scan produced 4 raster room candidates, no room-linked opening candidates,
and 4 separate wall-gap candidates. The room raster follows the dominant
cross-height wall orientation (about 27.3°); its polygons are not snapped to
rectangles. Candidate 04 has approximate oriented spans of 3.20 m × 3.44 m
and an estimated raster area of 8.08 m². These spans are numerically near some
of the measured side lengths, but there is no spatial registration between the
schematic reference and this scan region, so the association is unconfirmed.

The candidate outline is derived from a height-slice occupancy raster and
remains incomplete in places. Its area is not a reliable measurement. The
remeasured wall lengths do not determine ground-truth area without the full
perimeter and corner geometry.

The closest wall gap is 0.819 m, only 0.031 m (3.6%) narrower than the measured
doorway. Both bounding wall traces have support in all three height bands.
This is promising evidence, but the gap is not linked to a room pair: the
existing opening rule requires room candidates on both sides, while the scan
only has a nearby room candidate on one side. The scan and sketch also have not
been spatially registered, so this gap cannot yet be confirmed as this
doorway. The other wall-gap candidates measure 0.460, 0.466, and 0.570 m.

The 2.63 m ceiling height is not evaluated: the current output does not fit a
room-specific floor-to-ceiling height.

## What this establishes

- The scan contains room-shaped raster regions, including one with rough spans
  near some reported side lengths.
- The room-to-sketch match remains uncertain because the scan has no visual
  registration to the schematic reference and the full room shape is unknown.
- The scan contains a strongly supported wall gap close to the measured
  doorway width, but has not classified it as a doorway. The candidate room
  area is still substantially low.
- Record the room perimeter and corner geometry before treating area or shape
  as an accuracy metric. Then associate the measured room with a scan region
  and evaluate doorway and ceiling-height detection.
