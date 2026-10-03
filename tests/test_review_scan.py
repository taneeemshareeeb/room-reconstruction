from __future__ import annotations

import base64
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from PIL import Image

from scripts import review_scan


class CameraSessionEndpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self.temporary.name) / "camera_sessions"
        self.sessions_patch = patch.object(review_scan, "CAMERA_SESSIONS_DIR", self.sessions_dir)
        self.sessions_patch.start()
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            review_scan.make_handler(Path(self.temporary.name), app_mode=True, camera_mode=True),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.sessions_patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def image_data_url() -> str:
        image = Image.new("RGB", (32, 24), "#39765f")
        output = BytesIO()
        image.save(output, format="JPEG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def post_session(self, frames: list[dict[str, object]]) -> tuple[int, dict[str, object]]:
        body = json.dumps({"frames": frames}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/capture-sessions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urlopen(request)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def test_upload_list_and_inspect_captured_frame(self):
        status, payload = self.post_session([{
            "timestamp": "2026-10-04T12:00:00.000Z",
            "elapsed_ms": 0,
            "data_url": self.image_data_url(),
        }])
        self.assertEqual(status, 201)
        self.assertEqual(payload["frame_count"], 1)
        self.assertEqual(payload["reconstruction_status"], "not_processed")

        with urlopen(f"{self.base_url}/api/capture-sessions") as response:
            listed = json.loads(response.read())
        self.assertEqual(len(listed["sessions"]), 1)
        frame = listed["sessions"][0]["frames"][0]
        frame_url = f"{self.base_url}/api/capture-sessions/{payload['session_id']}/{frame['filename']}"
        with urlopen(frame_url) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get_content_type(), "image/jpeg")
            self.assertEqual(response.read()[:2], b"\xff\xd8")

    def test_rejects_invalid_image_without_creating_session(self):
        status, payload = self.post_session([{
            "timestamp": "2026-10-04T12:00:00+00:00",
            "elapsed_ms": 12,
            "data_url": "data:image/jpeg;base64,ZmFrZQ==",
        }])
        self.assertEqual(status, 400)
        self.assertIn("not a valid image", payload["error"])
        self.assertEqual(list(self.sessions_dir.glob("*/session.json")), [])


if __name__ == "__main__":
    unittest.main()
