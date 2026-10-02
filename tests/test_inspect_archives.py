import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.inspect_archives import inspect_archive, is_suspicious_path


class InspectArchiveTests(unittest.TestCase):
    def test_suspicious_paths(self):
        for path in ("../outside.txt", "folder/../../outside.txt", "/absolute.txt", "C:\\outside.txt"):
            with self.subTest(path=path):
                self.assertTrue(is_suspicious_path(path))
        for path in ("room/wall.obj", "folder/..notes.txt"):
            with self.subTest(path=path):
                self.assertFalse(is_suspicious_path(path))

    def test_archive_summary_uses_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("mesh/room.obj", "vertices")
                archive.writestr("readme", "info")
                archive.writestr("../outside", "unsafe")

            summary = inspect_archive(archive_path, sample_size=2)

        self.assertEqual(summary["entry_count"], 3)
        self.assertEqual(summary["total_uncompressed_size"], len("verticesinfounsafe"))
        self.assertEqual(summary["extensions"][".obj"], 1)
        self.assertEqual(summary["extensions"]["[no extension]"], 2)
        self.assertEqual(summary["sample_paths"], ["mesh/room.obj", "readme"])
        self.assertEqual(summary["suspicious_paths"], ["../outside"])


if __name__ == "__main__":
    unittest.main()
