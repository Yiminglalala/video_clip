import unittest
import os
from pathlib import Path

from src.project_paths import (
    DEFAULT_VIDEO_OUTPUT_DIR,
    OUTPUT_ROOT,
    PLAYWRIGHT_OUTPUT_DIR,
    QA_OUTPUT_DIR,
    SUBTITLE_OUTPUT_DIR,
    SUBTITLE_PROBE_DIR,
    VIDEO_OUTPUT_DIR,
)


class ProjectPathsTests(unittest.TestCase):
    def test_output_paths_are_classified_under_output_root(self):
        expected_video_dir = Path(
            os.environ.get("VIDEO_CLIP_VIDEO_OUTPUT_DIR", str(DEFAULT_VIDEO_OUTPUT_DIR))
        )
        self.assertEqual(VIDEO_OUTPUT_DIR, expected_video_dir)
        self.assertEqual(SUBTITLE_OUTPUT_DIR, OUTPUT_ROOT / "subtitles")
        self.assertEqual(QA_OUTPUT_DIR, OUTPUT_ROOT / "qa")
        self.assertEqual(PLAYWRIGHT_OUTPUT_DIR, QA_OUTPUT_DIR / "playwright")
        self.assertEqual(SUBTITLE_PROBE_DIR, QA_OUTPUT_DIR / "subtitle_probe")


if __name__ == "__main__":
    unittest.main()
