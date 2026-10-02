# Scan geometry candidate comparison

Both captures were sampled with every 10th frame and 2 cm voxels. The point
clouds were read directly from their ZIP archives; the ZIP files were not
extracted or modified. The same floor-height, wall-line, enclosed-region, and
opening-gap heuristics were applied to each cloud.

| Capture | Point count | Estimated floor y (m) | Wall candidates | Room candidates | Opening candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Floor only | 1,348,192 | -1.409 | 25 | 3 | 1 |
| Floor and ceiling | 2,707,543 | -1.494 | 21 | 4 | 0 |

Candidate room areas were 4.17, 19.23, and 5.36 m² for the floor-only capture;
the 19.23 m² region may include a hall or multiple connected spaces. The
floor-and-ceiling capture produced 5.95, 3.79, 7.26, and 7.05 m² candidates.
The floor-only scan produced one possible 0.723 m opening gap between two room
candidates. It may be a doorway or missing scan data.

These counts do not measure accuracy. There is no ground-truth plan yet, and
the candidate counts change with the scan and the gap-closing settings. The
comparison shows that adding the ceiling-inclusive data did not automatically
produce a more connected room layout in the current pipeline.

## Review images

- [Floor-only wall candidates](single_scan_floor_only_wall_candidates.png)
- [Floor-only room and opening candidates](single_scan_floor_only_room_candidates.png)
- [Floor-and-ceiling wall candidates](single_scan_with_ceiling_wall_candidates.png)
- [Floor-and-ceiling room candidates](single_scan_with_ceiling_room_candidates.png)
- [Height slices from the floor-and-ceiling capture](single_scan_with_ceiling_height_slices.png)
