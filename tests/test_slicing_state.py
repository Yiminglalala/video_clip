import unittest

from src.slicing_state import (
    apply_segment_edit,
    rebuild_flat_segments_from_export_segments,
    reset_segment_edits,
    reset_slicing_workflow_state,
    sync_ui_segments_to_export_segments,
)


class DummySong:
    def __init__(self, title="old", artist="old_artist"):
        self.song_title = title
        self.song_artist = artist


class DummySegment:
    confidence = 0.8
    song_index = 2


class SlicingStateTests(unittest.TestCase):
    def test_rebuild_flat_segments_preserves_export_fields(self):
        flat = rebuild_flat_segments_from_export_segments(
            [
                {
                    "start": 1,
                    "end": 9,
                    "type": "副歌",
                    "songformer_label": "chorus",
                    "is_highlight": True,
                    "segment": DummySegment(),
                }
            ]
        )
        self.assertEqual(flat[0]["current_label"], "副歌")
        self.assertEqual(flat[0]["songformer_label"], "chorus")
        self.assertEqual(flat[0]["song_index"], 2)
        self.assertTrue(flat[0]["is_highlight"])

    def test_apply_segment_edit_keeps_neighbors_contiguous(self):
        segments = [
            {"start": 0.0, "end": 10.0, "current_label": "主歌", "songformer_label": "verse"},
            {"start": 10.0, "end": 20.0, "current_label": "主歌", "songformer_label": "verse"},
            {"start": 20.0, "end": 30.0, "current_label": "副歌", "songformer_label": "chorus"},
        ]
        changed = apply_segment_edit(segments, 1, "副歌", 12.0, 22.0, "chorus")
        self.assertTrue(changed)
        self.assertEqual(segments[0]["end"], 12.0)
        self.assertEqual(segments[2]["start"], 22.0)
        self.assertEqual(segments[1]["current_label"], "副歌")

    def test_sync_ui_segments_to_export_segments_updates_song_name(self):
        song = DummySong()
        export_segments = [{"start": 0, "end": 10, "type": "主歌", "songformer_label": "verse", "song": song}]
        ui_segments = [{"start": 1.5, "end": 11.5, "current_label": "副歌", "songformer_label": "chorus", "song_index": 3}]
        songs_info = [{"song_index": 3, "song_title": "新歌名", "song_artist": "新歌手"}]

        sync_ui_segments_to_export_segments(ui_segments, export_segments, songs_info)

        self.assertEqual(export_segments[0]["start"], 1.5)
        self.assertEqual(export_segments[0]["type"], "副歌")
        self.assertEqual(export_segments[0]["songformer_label"], "chorus")
        self.assertEqual(song.song_title, "新歌名")
        self.assertEqual(song.song_artist, "新歌手")

    def test_reset_segment_edits_restores_initial_label_only(self):
        segments = [{"current_label": "副歌", "_initial_label": "主歌", "modified": True, "start": 1.0}]
        reset_segment_edits(segments)
        self.assertEqual(segments[0]["current_label"], "主歌")
        self.assertFalse(segments[0]["modified"])
        self.assertEqual(segments[0]["start"], 1.0)

    def test_reset_slicing_workflow_state_clears_preview_keys(self):
        state = {"slice_preview_1_path": "x", "slice_segments": [{"x": 1}]}
        reset_slicing_workflow_state(state)
        self.assertNotIn("slice_preview_1_path", state)
        self.assertEqual(state["slice_workflow_step"], 0)
        self.assertEqual(state["slice_segments"], [])


if __name__ == "__main__":
    unittest.main()
