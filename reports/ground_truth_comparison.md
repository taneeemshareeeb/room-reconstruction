# Room layout comparison

This comparison uses the manually measured layout in
[`data/ground_truth/room_layout.json`](../data/ground_truth/room_layout.json)
and the current ceiling-inclusive run produced with frame step 10 and 2 cm
voxels. The sketch is schematic and explicitly flags both room lengths for
rechecking.

## Measurements

| Measurement | Value | Reading |
| --- | ---: | --- |
| Top width | 3.36 m | Recorded in sketch |
| Bottom width | 3.32 m | Recorded in sketch |
| Left length | 3.63 m | Check; not confirmed |
| Right length | 4.15 m | Check; not confirmed |
| Ceiling height | 2.63 m | Confirmed by user |
| Doorway width | 0.85 m | Confirmed by user; bottom wall near right corner |

The two widths agree within 0.04 m. The side lengths disagree by 0.52 m, so
there is no single trusted room depth or area yet.

## Current scan comparison

The scan produced 4 room candidates, no room-linked opening candidates, and 4
separate wall-gap candidates. Candidate 03 is
the closest size match by its outline's approximate rotated spans, about
3.22 m × 3.93 m. Against the measured width average of 3.34 m, its short span
is about 0.12 m (3.6%) smaller. Its long span is between the two conflicting
measurements: 0.30 m longer than 3.63 m or 0.22 m shorter than 4.15 m. This is
a possible match, not a confirmed association between the sketch and that part
of the scan.

Candidate 03 reports 7.26 m². Multiplying the average width by either disputed
length gives 12.12–13.86 m², but that range is only a rectangular estimate and
cannot serve as settled ground truth. The candidate area is substantially
smaller and its outline is fragmented, so the current room boundary and area
are not reliable measurements.

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

- The scan contains a room candidate with rough dimensions in the right range.
- The room-to-sketch match remains uncertain because the scan has no visual
  registration to the sketch and the room lengths are unresolved.
- The scan contains a strongly supported wall gap close to the measured
  doorway width, but has not classified it as a doorway. The candidate room
  area is still substantially low.
- Re-measure the left and right room lengths before treating area or depth as
  an accuracy metric. Then associate the measured room with a scan region and
  evaluate doorway and ceiling-height detection.
