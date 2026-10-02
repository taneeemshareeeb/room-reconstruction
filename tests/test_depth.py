import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from room_reconstruction.depth import load_depth_frame
from room_reconstruction.pointcloud import (
    load_archive_point_cloud,
    load_capture_point_cloud,
    voxel_downsample,
)


class LoadDepthFrameTests(unittest.TestCase):
    def _make_capture(self, root, depth, confidence, row):
        for directory in ("depth", "confidence"):
            (root / directory).mkdir()
        Image.fromarray(depth.astype(np.uint16)).save(root / "depth" / "000007.png")
        Image.fromarray(confidence.astype(np.uint8)).save(
            root / "confidence" / "000007.png"
        )
        with (root / "odometry.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_projection_mask_and_world_transform(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            depth = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
            confidence = np.array([[2, 1], [2, 2]], dtype=np.uint8)
            row = {
                "frame": "000007",
                "x": "10", "y": "20", "z": "30",
                "qx": "0", "qy": "0", "qz": "0.7071067811865476",
                "qw": "0.7071067811865476",
                # With a 2x2 depth image and 20x20 RGB calibration, the scaled
                # intrinsics are fx=fy=2 and cx=cy=1.
                "fx": "20", "fy": "20", "cx": "10", "cy": "10",
            }
            self._make_capture(root, depth, confidence, row)

            points = load_depth_frame(
                root,
                7,
                rgb_size=(20, 20),
                min_confidence=2,
                min_depth_m=0.1,
                max_depth_m=3.5,
            )

        expected = np.array(
            [[10.5, 19.5, 31.0], [10.0, 18.5, 33.0]], dtype=np.float64
        )
        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(points, expected)
        self.assertTrue(np.isfinite(points).all())

    def test_multiframe_selection_and_voxel_downsampling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            depth = np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16)
            confidence = np.array([[2, 1], [2, 2]], dtype=np.uint8)
            first = {
                "frame": "000007",
                "x": "0", "y": "0", "z": "0",
                "qx": "0", "qy": "0", "qz": "0", "qw": "1",
                "fx": "20", "fy": "20", "cx": "10", "cy": "10",
            }
            second = {**first, "frame": "000008", "x": "1"}
            for directory in ("depth", "confidence"):
                (root / directory).mkdir()
            for frame_id in (7, 8):
                name = f"{frame_id:06d}.png"
                Image.fromarray(depth).save(root / "depth" / name)
                Image.fromarray(confidence).save(root / "confidence" / name)
            with (root / "odometry.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(first))
                writer.writeheader()
                writer.writerows((first, second))

            all_frames = load_capture_point_cloud(
                root,
                frame_step=1,
                rgb_size=(20, 20),
                max_depth_m=3.5,
                voxel_size=0.01,
            )
            every_other = load_capture_point_cloud(
                root,
                frame_step=2,
                rgb_size=(20, 20),
                max_depth_m=3.5,
                voxel_size=0.01,
            )
            reduced = voxel_downsample(
                np.vstack((every_other, every_other)), voxel_size=0.01
            )
            archive_path = root / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(root / "odometry.csv", "capture/odometry.csv")
                for frame_id in (7, 8):
                    name = f"{frame_id:06d}.png"
                    for directory in ("depth", "confidence"):
                        archive.write(root / directory / name, f"capture/{directory}/{name}")
            from_archive = load_archive_point_cloud(
                archive_path,
                frame_step=1,
                rgb_size=(20, 20),
                max_depth_m=3.5,
                voxel_size=0.01,
                chunk_frames=1,
            )

        self.assertEqual(all_frames.shape, (4, 3))
        self.assertEqual(every_other.shape, (2, 3))
        np.testing.assert_array_equal(reduced, every_other)
        np.testing.assert_allclose(from_archive, all_frames)
        self.assertTrue(np.all(all_frames[2:, 0] > all_frames[:2, 0]))


if __name__ == "__main__":
    unittest.main()
