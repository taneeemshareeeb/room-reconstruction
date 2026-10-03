"""Tests for the image-space photo/video capture tiers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from room_reconstruction.image_tiers import process_image_tier, visual_plan_document


class ImageTierTests(unittest.TestCase):
    def test_photos_write_overlap_diagnostics_and_non_metric_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "photos"
            capture.mkdir()
            rng = np.random.default_rng(20)
            first = np.full((420, 640, 3), 245, dtype=np.uint8)
            for _ in range(180):
                center = (int(rng.integers(5, 635)), int(rng.integers(5, 415)))
                color = tuple(int(value) for value in rng.integers(0, 220, 3))
                cv2.circle(first, center, int(rng.integers(2, 8)), color, -1)
            second = cv2.warpAffine(first, np.float32([[1, 0, -35], [0, 1, 0]]), (640, 420))
            Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)).save(capture / "01.png")
            Image.fromarray(cv2.cvtColor(second, cv2.COLOR_BGR2RGB)).save(capture / "02.png")

            output = root / "result"
            report = process_image_tier(capture, tier="photos", output_dir=output)
            plan = visual_plan_document(tier="photos", capture="photos", image_analysis=report)

            self.assertEqual(report["image_count"], 2)
            self.assertEqual(len(report["adjacent_pair_matches"]), 1)
            self.assertTrue((output / "plan.png").is_file())
            self.assertTrue((output / "image_analysis.json").is_file())
            self.assertEqual(plan["rooms"], [])
            self.assertEqual(plan["capture"]["tier"], "photos")
            self.assertTrue(plan["limitations"])
            json.loads((output / "image_analysis.json").read_text())

    def test_photo_tier_reports_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "no supported photos inputs"):
                process_image_tier(temporary, tier="photos", output_dir=Path(temporary) / "out")


if __name__ == "__main__":
    unittest.main()
