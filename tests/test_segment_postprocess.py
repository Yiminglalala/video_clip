# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

from src.segment_postprocess import normalize_export_segment_overlaps


class SegmentPostprocessTests(unittest.TestCase):
    def _song(self):
        return SimpleNamespace(song_index=0, start_time=0.0)

    def _assert_no_overlap(self, segments):
        for left, right in zip(segments, segments[1:]):
            self.assertLessEqual(float(left["end"]), float(right["start"]) + 1e-9)

    def test_overlap_uses_sentence_boundary_when_duration_allows(self):
        song = self._song()
        segments = [
            {"song": song, "start": 190.9, "end": 203.9, "type": "副歌"},
            {"song": song, "start": 196.9, "end": 214.3, "type": "合唱"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 192.0, "end": 196.4, "text": "s1"},
                    {"start": 197.0, "end": 202.8, "text": "s2"},
                    {"start": 203.2, "end": 207.5, "text": "s3"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 8.0, 18.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], fixed[1]["start"], places=3)
        self.assertGreaterEqual(fixed[0]["end"] - fixed[0]["start"], 8.0)
        self.assertGreaterEqual(fixed[1]["end"] - fixed[1]["start"], 8.0)

    def test_impossible_min_duration_drops_short_lower_priority_segment(self):
        song = self._song()
        segments = [
            {"song": song, "start": 190.9, "end": 203.9, "type": "副歌"},
            {"song": song, "start": 196.9, "end": 214.3, "type": "合唱"},
        ]
        cached_asr = {0: {"sentences": [{"start": 198.0, "end": 204.5, "text": "shared"}]}}

        fixed = normalize_export_segment_overlaps(segments, 12.0, 18.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertTrue(fixed)
        self.assertTrue(all(float(item["end"]) - float(item["start"]) >= 12.0 for item in fixed))

    def test_no_subtitle_falls_back_to_midpoint_without_overlap(self):
        song = self._song()
        segments = [
            {"song": song, "start": 10.0, "end": 25.0, "type": "主歌"},
            {"song": song, "start": 20.0, "end": 38.0, "type": "副歌"},
        ]

        fixed = normalize_export_segment_overlaps(segments, 8.0, 20.0, cached_asr_results={})

        self._assert_no_overlap(fixed)
        self.assertTrue(all(float(item["end"]) - float(item["start"]) >= 8.0 for item in fixed))


if __name__ == "__main__":
    unittest.main()
