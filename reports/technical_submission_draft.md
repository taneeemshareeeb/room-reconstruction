# Technical Submission Draft

## 1. Summary

This repository implements a partial room-reconstruction pipeline for phone LiDAR-style captures. The verified current state is a candidate-generation system that produces a sparse world-space point cloud, wall-trace hypotheses, cross-height wall support, and partial room-region suggestions. It does not claim a validated connected floor plan or confirmed openings.

The implementation is intentionally conservative: output metadata and JSON files label results as partial, candidate-only, and unverified. The code and outputs are therefore better described as a milestone pipeline rather than a complete reconstruction system.

## 2. Scope of work and architecture

The project is organized around a modular candidate pipeline:

- data loading and depth projection from capture metadata
- voxel downsampling and world-coordinate point-cloud assembly
- pass detection for scan segments with distinct pitch behavior
- wall trace detection in a horizontal slice near the floor height
- cross-height agreement analysis for wall support consistency
- room candidate generation from enclosed regions
- opening and wall-gap inference from wall traces and nearby support
- visual preview rendering for review and inspection

The main package entry point is run.py, and the CLI orchestration is implemented in src/room_reconstruction/cli.py. The output document is assembled in src/room_reconstruction/plan.py and is deliberately structured to report partial status and uncertainty rather than claim metric correctness.

## 3. Experiments actually performed

The following actions were executed in the current workspace and are the basis for the submission status:

1. Project setup and dependency installation in the local virtual environment.
2. Execution of the repository unit test suite:
   - command: python -m pytest -q
   - result: 11 passed, 6 subtests passed
3. Archive inspection on the raw capture data using the project script for ZIP metadata review.
4. Execution of the LiDAR CLI on the sample capture:
   - command: PYTHONPATH=src python run.py data/raw/single_room.zip --tier lidar --frame-step 10 --voxel-size 0.02 --output-dir outputs/single_room
   - result: output files were generated successfully
5. Inspection of the generated output document in outputs/single_room/plan.json.
6. Review of the generated artifact set under outputs/single_room and confirmation of partial candidate outputs.

## 4. Verified results

The tested and generated output confirms a partial pipeline state.

From the verified output at outputs/single_room/plan.json:

- status: partial
- point_count: 389816
- wall_candidates: 6
- cross_height_wall_candidates: 4
- room_candidates: 0
- opening_candidates: 0

This result indicates that the system can build a meaningful point cloud and extract candidate wall evidence, but it does not yet produce a validated room layout or opening map.

## 5. Interpretation of the evidence

The verified evidence is sufficient to support the following claims:

- the pipeline runs successfully in the current environment
- the project writes structured artifact outputs
- wall-like candidate traces are generated from the scan geometry
- cross-height agreement is detected across multiple horizontal bands
- room and opening hypotheses are emitted as candidate-only signals, not confirmed measurements

The evidence is not sufficient to support stronger claims such as:

- accurate connected room topology
- validated doorway or window width detections
- exact room area or wall-length measurements
- reliable ceiling-to-room assignment
- end-to-end metric reconstruction correctness

## 6. Known limitations and risk areas

The current implementation explicitly carries several limitations:

- no ground-truth room layout is available for the sample capture data
- room connectivity is not validated
- wall gaps are treated as geometric evidence and not confirmed openings
- ceiling peaks may correspond to ceilings, beams, or headers rather than true room boundaries
- the pipeline is sensitive to drift, missing walls, mirror-like surfaces, and sparse coverage
- generated wall and room outlines should be treated as review artifacts rather than final measurements

## 7. Not evaluated

The following items were not validated in this repository state and should be marked as not evaluated rather than assumed:

- calibrated room-to-room connectivity
- opening accuracy against measured door and window widths
- room-level ceiling assignment
- full drift correction and ablation results on large scans
- whole-house ground-truth comparison
- end-to-end real-world photo and video validation against a trusted reference

## 8. Submission posture

This repository is best submitted as a milestone prototype and candidate-generation pipeline, not as a complete reconstruction system. The code and outputs are honest about uncertainty, and the verified artifacts support the claim that the project is functional at the candidate detection stage while remaining incomplete at the verified room-layout stage.
