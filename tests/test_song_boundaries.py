# -*- coding: utf-8 -*-

import unittest

from src.processor import LiveVideoProcessor


class SongBoundaryTests(unittest.TestCase):
    def _processor_with_duration(self, duration: float) -> LiveVideoProcessor:
        processor = LiveVideoProcessor.__new__(LiveVideoProcessor)
        processor._get_audio_duration = lambda _audio_path: duration
        return processor

    def test_msaf_boundaries_do_not_split_song_scope(self):
        processor = self._processor_with_duration(540.693333)

        songs = processor._merge_boundaries(
            video_boundaries=[],
            audio_boundaries=[],
            msaf_boundaries=[120.0, 285.2803628117914, 503.8265759637188],
            audio_path="dummy.wav",
        )

        self.assertEqual(songs, [(0.0, 540.693333)])

    def test_ocr_boundaries_still_split_song_scope(self):
        processor = self._processor_with_duration(540.0)

        songs = processor._merge_boundaries(
            video_boundaries=[180.0, 360.0],
            audio_boundaries=[],
            msaf_boundaries=[285.0],
            audio_path="dummy.wav",
        )

        self.assertEqual(songs, [(0.0, 180.0), (180.0, 360.0), (360.0, 540.0)])


if __name__ == "__main__":
    unittest.main()
