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

    def test_adjacent_boundary_inside_sentence_moves_to_sentence_end(self):
        song = self._song()
        segments = [
            {"song": song, "start": 52.62, "end": 65.62, "type": "涓绘瓕"},
            {"song": song, "start": 65.62, "end": 80.16, "type": "鍓瓕"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 64.80, "end": 66.20, "text": "来呀来呀来晒秋"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 8.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 66.55, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 66.55, places=3)

    def test_adjacent_boundary_inside_sentence_can_assign_to_next(self):
        song = self._song()
        segments = [
            {"song": song, "start": 52.62, "end": 65.62, "type": "涓绘瓕"},
            {"song": song, "start": 65.62, "end": 80.16, "type": "鍓瓕"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 65.50, "end": 67.20, "text": "来呀来呀来晒秋"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 8.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 65.50, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 65.50, places=3)

    def test_adjacent_boundary_at_sentence_end_gets_tail_padding(self):
        song = self._song()
        segments = [
            {"song": song, "start": 52.62, "end": 65.62, "type": "涓绘瓕"},
            {"song": song, "start": 65.62, "end": 80.16, "type": "鍓瓕"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 64.80, "end": 65.62, "text": "来呀来呀来晒秋"},
                    {"start": 67.00, "end": 69.00, "text": "下一句"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 8.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 65.97, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 65.97, places=3)

    def test_activity_timestamps_do_not_change_boundary_repair(self):
        song = self._song()
        segments = [
            {"song": song, "start": 33.84, "end": 52.97, "type": "涓绘瓕"},
            {"song": song, "start": 52.97, "end": 64.70, "type": "涓绘瓕"},
            {"song": song, "start": 64.70, "end": 80.16, "type": "鍓瓕"},
            {"song": song, "start": 80.16, "end": 95.56, "type": "鍓瓕"},
        ]

        fixed_with_activity = normalize_export_segment_overlaps(
            segments,
            13.0,
            18.0,
            cached_asr_results={},
            activity_timestamps=[(53.04, 68.00), (68.01, 79.30)],
        )
        fixed_without_activity = normalize_export_segment_overlaps(
            segments,
            13.0,
            18.0,
            cached_asr_results={},
        )

        self._assert_no_overlap(fixed_with_activity)
        self.assertEqual(
            [(item["start"], item["end"]) for item in fixed_with_activity],
            [(item["start"], item["end"]) for item in fixed_without_activity],
        )
        self.assertTrue(all(float(item["end"]) - float(item["start"]) >= 13.0 - 1e-6 for item in fixed_with_activity))

    def test_final_sentence_validation_runs_after_duration_repair(self):
        song = self._song()
        segments = [
            {"song": song, "start": 177.96, "end": 190.00, "type": "副歌"},
            {"song": song, "start": 190.00, "end": 205.09, "type": "副歌"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 188.50, "end": 193.40, "text": "不能切断的歌词句子"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(
            segments,
            13.0,
            18.0,
            cached_asr_results=cached_asr,
            activity_timestamps=[(171.49, 191.02), (191.03, 197.40)],
        )

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 193.40, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 193.40, places=3)

    def test_existing_boundary_from_aed_end_is_rechecked_against_asr_sentence(self):
        song = self._song()
        segments = [
            {"song": song, "start": 222.21, "end": 238.06, "type": "主歌"},
            {"song": song, "start": 238.06, "end": 256.06, "type": "副歌"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 235.20, "end": 240.20, "text": "跨过当前切点的歌词句子"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(
            segments,
            13.0,
            18.0,
            cached_asr_results=cached_asr,
            activity_timestamps=[(222.22, 238.06), (240.05, 250.06)],
        )

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 240.55, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 240.55, places=3)

    def test_lyrics_covered_short_segments_are_kept_instead_of_dropped(self):
        song = self._song()
        segments = [
            {"song": song, "start": 85.96, "end": 100.43, "type": "副歌"},
            {"song": song, "start": 100.43, "end": 112.81, "type": "副歌"},
            {"song": song, "start": 112.81, "end": 123.75, "type": "副歌"},
            {"song": song, "start": 123.75, "end": 136.72, "type": "副歌"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 101.32, "end": 106.30, "text": "让幸福眼泪自由的脸庞"},
                    {"start": 107.48, "end": 112.46, "text": "轻轻追寻风中歌声的方向"},
                    {"start": 113.68, "end": 119.70, "text": "我要向那遥远遥远的地方"},
                    {"start": 121.04, "end": 123.46, "text": "扎西德勒"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 13.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertEqual(len(fixed), 4)
        self.assertLessEqual(max(float(r["start"]) - float(l["end"]) for l, r in zip(fixed, fixed[1:])), 0.5)
        self.assertTrue(any(float(item["start"]) <= 101.32 and float(item["end"]) >= 112.46 for item in fixed))
        self.assertTrue(any(float(item["start"]) <= 113.68 and float(item["end"]) >= 123.46 for item in fixed))

    def test_large_gap_with_asr_coverage_is_filled(self):
        song = self._song()
        segments = [
            {"song": song, "start": 85.96, "end": 100.43, "type": "副歌"},
            {"song": song, "start": 123.75, "end": 136.72, "type": "副歌"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 101.32, "end": 106.30, "text": "让幸福眼泪自由的脸庞"},
                    {"start": 107.48, "end": 112.46, "text": "轻轻追寻风中歌声的方向"},
                    {"start": 113.68, "end": 119.70, "text": "我要向那遥远遥远的地方"},
                    {"start": 121.04, "end": 123.46, "text": "扎西德勒"},
                ]
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 13.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertGreater(len(fixed), 2)
        self.assertLessEqual(max(float(r["start"]) - float(l["end"]) for l, r in zip(fixed, fixed[1:])), 0.5)
        self.assertTrue(any(item.get("filled_from_asr_gap") for item in fixed))


    def test_word_progress_assigns_barely_started_sentence_to_next(self):
        song = self._song()
        segments = [
            {"song": song, "start": 52.62, "end": 65.62, "type": "verse"},
            {"song": song, "start": 65.62, "end": 80.16, "type": "chorus"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 64.80, "end": 68.20, "text": "abcdef"},
                ],
                "words": [
                    {"start": 64.80, "end": 65.00, "word": "a"},
                    {"start": 65.00, "end": 65.20, "word": "b"},
                    {"start": 65.20, "end": 65.50, "word": "c"},
                    {"start": 65.50, "end": 66.10, "word": "d"},
                    {"start": 66.10, "end": 67.00, "word": "e"},
                    {"start": 67.00, "end": 68.20, "word": "f"},
                ],
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 8.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 64.80, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 64.80, places=3)

    def test_word_progress_assigns_mostly_sung_sentence_to_prev(self):
        song = self._song()
        segments = [
            {"song": song, "start": 52.62, "end": 67.20, "type": "verse"},
            {"song": song, "start": 67.20, "end": 82.16, "type": "chorus"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 64.80, "end": 68.20, "text": "abcdef"},
                ],
                "words": [
                    {"start": 64.80, "end": 65.00, "word": "a"},
                    {"start": 65.00, "end": 65.20, "word": "b"},
                    {"start": 65.20, "end": 65.50, "word": "c"},
                    {"start": 65.50, "end": 66.10, "word": "d"},
                    {"start": 66.10, "end": 67.00, "word": "e"},
                    {"start": 67.00, "end": 68.20, "word": "f"},
                ],
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 8.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 68.55, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 68.55, places=3)

    def test_word_progress_uses_text_units_for_multi_char_tokens(self):
        song = self._song()
        segments = [
            {"song": song, "start": 52.62, "end": 65.62, "type": "verse"},
            {"song": song, "start": 65.62, "end": 80.16, "type": "chorus"},
        ]
        cached_asr = {
            0: {
                "sentences": [
                    {"start": 64.80, "end": 68.20, "text": "abcdef"},
                ],
                "words": [
                    {"start": 64.80, "end": 66.00, "word": "abc"},
                    {"start": 66.00, "end": 68.20, "word": "def"},
                ],
            }
        }

        fixed = normalize_export_segment_overlaps(segments, 8.0, 15.0, cached_asr_results=cached_asr)

        self._assert_no_overlap(fixed)
        self.assertAlmostEqual(fixed[0]["end"], 64.80, places=3)
        self.assertAlmostEqual(fixed[1]["start"], 64.80, places=3)


if __name__ == "__main__":
    unittest.main()
