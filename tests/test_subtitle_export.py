# -*- coding: utf-8 -*-

import unittest

from src.subtitle_export import DEFAULT_SUBTITLE_MAX_CHARS, split_subtitle_sentence_entry


class SubtitleExportTests(unittest.TestCase):
    def test_long_sentence_splits_into_max_10_char_chunks(self):
        sentence = {
            "text": "\u6211\u5df2\u77ed\u6682\u7231\u4f60\u4e0d\u4f11\u7684\u54ed\u5b8c\u4e4b\u540e\u8fd8\u662f\u60f3\u4f60",
            "start": 0.0,
            "end": 6.0,
        }

        chunks = split_subtitle_sentence_entry(sentence)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(item["text"]) <= DEFAULT_SUBTITLE_MAX_CHARS for item in chunks))
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[-1]["end"], 6.0)

    def test_balanced_split_avoids_single_character_tail(self):
        sentence = {
            "text": "\u6211\u5df2\u77ed\u6682\u7231\u4f60\u4e0d\u4f11\u7684\u54ed\u5b8c",  # 11 chars
            "start": 0.0,
            "end": 3.0,
        }

        chunks = split_subtitle_sentence_entry(sentence)

        lengths = [len(item["text"]) for item in chunks]
        self.assertEqual(lengths, [6, 5])
        self.assertTrue(all(length >= 4 for length in lengths))


if __name__ == "__main__":
    unittest.main()
