"""Exercise real video encoding, frame identity and inclusive cut boundaries."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import hand_step_pipeline as hand_steps


class ExactFrameCutTests(unittest.TestCase):
    def test_source_frames_survive_integer_and_fractional_fps(self):
        if not shutil.which(os.environ.get("FFMPEG_BIN", "ffmpeg")):
            self.skipTest("FFMPEG_BIN/ffmpeg required for real encoding regression")
        with tempfile.TemporaryDirectory() as tmp:
            for fps in (24.0, 25.0, 30000 / 1001, 30.0, 60.0):
                with self.subTest(fps=fps):
                    source = str(Path(tmp) / f"source-{fps}.mp4")
                    output = str(Path(tmp) / "clip.mp4")
                    writer = cv2.VideoWriter(
                        source, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64)
                    )
                    self.assertTrue(writer.isOpened())
                    for frame in range(60):
                        writer.write(np.full((64, 64, 3), 10 + frame * 37 % 230, np.uint8))
                    writer.release()

                    def decode(path):
                        cap = cv2.VideoCapture(path)
                        means = []
                        rate = cap.get(cv2.CAP_PROP_FPS)
                        while True:
                            ok, image = cap.read()
                            if not ok:
                                break
                            means.append(float(image.mean()))
                        cap.release()
                        return means, rate

                    expected, _ = decode(source)
                    for start, end in ((10, 33), (59, 59)):
                        hand_steps._cut_exact_source_frames(
                            video_path=source, output_path=output,
                            start_frame_id=start, end_frame_id=end, fps=fps,
                        )
                        actual, rate = decode(output)
                        self.assertEqual(len(actual), end - start + 1)
                        self.assertAlmostEqual(rate, fps, delta=0.01)
                        np.testing.assert_allclose(actual, expected[start:end + 1], atol=4, rtol=0)

    def test_invalid_fps_rejected_before_encoding(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                hand_steps._cut_exact_source_frames(
                    video_path="unused", output_path="unused",
                    start_frame_id=0, end_frame_id=1, fps=fps,
                )


if __name__ == "__main__":
    unittest.main()
