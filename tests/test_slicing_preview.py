import unittest

from src.output_spec import OutputResolutionSpec
from src.slicing_preview import build_preview_cache_key, find_song_start_time


class DummySong:
    def __init__(self, song_index, start_time):
        self.song_index = song_index
        self.start_time = start_time


class SlicingPreviewTests(unittest.TestCase):
    def test_cache_key_changes_with_time_and_output_spec(self):
        spec_a = OutputResolutionSpec(1080, 1920, "portrait", "1080x1920")
        spec_b = OutputResolutionSpec(1920, 1080, "landscape", "1920x1080")
        seg = {"start": 1.234, "end": 9.876}

        self.assertNotEqual(
            build_preview_cache_key(1, seg, spec_a),
            build_preview_cache_key(1, seg, spec_b),
        )
        self.assertIn("1.23_9.88", build_preview_cache_key(1, seg, spec_a))

    def test_find_song_start_time_falls_back_to_zero(self):
        songs = [DummySong(2, 30.5)]
        self.assertEqual(find_song_start_time(songs, 2), 30.5)
        self.assertEqual(find_song_start_time(songs, 9), 0.0)


if __name__ == "__main__":
    unittest.main()
