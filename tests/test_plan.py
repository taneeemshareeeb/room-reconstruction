import contextlib
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from room_reconstruction.cli import main
from room_reconstruction.geometry import detect_wall_candidates
from room_reconstruction.plan import SCHEMA_VERSION, build_plan_document
from room_reconstruction.rooms import detect_room_candidates


class PlanDocumentTests(unittest.TestCase):
    def test_detects_unverified_wall_traces_in_synthetic_room(self):
        heights = np.arange(0, 2.51, 0.05)
        x = np.arange(0, 4.01, 0.05)
        z = np.arange(0, 3.01, 0.05)
        walls = [
            np.array([(0.0, y, horizontal) for horizontal in z for y in heights]),
            np.array([(4.0, y, horizontal) for horizontal in z for y in heights]),
            np.array([(horizontal, y, 0.0) for horizontal in x for y in heights]),
            np.array([(horizontal, y, 3.0) for horizontal in x for y in heights]),
        ]

        candidates = detect_wall_candidates(np.concatenate(walls))
        lengths = sorted(candidate["length_m"] for candidate in candidates)

        self.assertGreaterEqual(len(candidates), 4)
        self.assertTrue(all(candidate["status"] == "candidate_unverified" for candidate in candidates))
        self.assertTrue(any(abs(length - 4.0) < 0.2 for length in lengths))
        self.assertTrue(any(2.5 < length < 3.2 for length in lengths))

    def test_finds_two_enclosed_room_candidates(self):
        heights = np.arange(0, 2.51, 0.05)
        points = []
        for x0 in (0.0, 5.0):
            xs = np.arange(x0, x0 + 4.01, 0.05)
            zs = np.arange(0, 3.01, 0.05)
            for x in (x0, x0 + 4.0):
                points.extend((x, y, z) for z in zs for y in heights)
            for z in (0.0, 3.0):
                points.extend((x, y, z) for x in xs for y in heights)

        candidates = detect_room_candidates(np.asarray(points))

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["status"] == "candidate_unverified" for candidate in candidates))
        self.assertTrue(all(2.5 < candidate["estimated_area_m2"] < 12 for candidate in candidates))
        self.assertTrue(all(candidate["boundary_xz_m"][0] == candidate["boundary_xz_m"][-1]
                            for candidate in candidates))

    def test_first_milestone_is_explicitly_partial(self):
        result = build_plan_document(
            capture="sample", point_count=12, frame_step=10, voxel_size_m=0.02
        )

        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["rooms"], [])
        self.assertEqual(result["processing"]["point_count"], 12)
        self.assertEqual(result["capabilities"]["floor_area"], "not_implemented")
        self.assertIn("not a floor plan", result["limitations"][0].lower())

    def test_runner_writes_partial_result_and_preview(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            capture = root / "capture.zip"
            output = root / "result"
            row = {
                "frame": "000001", "x": "0", "y": "0", "z": "0",
                "qx": "0", "qy": "0", "qz": "0", "qw": "1",
                "fx": "20", "fy": "20", "cx": "10", "cy": "10",
            }
            with zipfile.ZipFile(capture, "w") as archive:
                stream = io.StringIO()
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
                archive.writestr("capture/odometry.csv", stream.getvalue())
                for folder, dtype in (("depth", np.uint16), ("confidence", np.uint8)):
                    image = np.full((2, 2), 1000 if folder == "depth" else 2, dtype=dtype)
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="PNG")
                    archive.writestr(f"capture/{folder}/000001.png", buffer.getvalue())

            with contextlib.redirect_stdout(io.StringIO()):
                result = main([str(capture), "--frame-step", "1", "--output-dir", str(output)])

            self.assertEqual(result, 0)
            document = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "partial")
            self.assertEqual(document["processing"]["point_count"], 4)
            self.assertEqual(document["rooms"], [])
            self.assertIn("room_candidates", document)
            self.assertTrue((output / "plan.png").is_file())
            self.assertTrue((output / "wall_candidates.png").is_file())
            self.assertTrue((output / "wall_candidates.json").is_file())
            self.assertTrue((output / "room_candidates.png").is_file())
            self.assertTrue((output / "room_candidates.json").is_file())
            with np.load(output / "point_cloud.npz") as point_cloud:
                self.assertEqual(point_cloud["points"].shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
