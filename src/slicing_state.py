"""State helpers for the Streamlit slicing workflow.

These helpers keep UI edits and export payloads synchronized in one place.
They intentionally avoid importing Streamlit so they are easy to unit test.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence


SLICE_STATE_DEFAULTS: dict[str, Any] = {
    "slice_workflow_step": 0,
    "slice_video_source": None,
    "slice_analysis_result": None,
    "slice_output_files": None,
    "slice_export_segments": None,
    "slice_segments": [],
    "slice_selected_segment": None,
    "slice_jump_to_time": None,
    "slice_auto_play_segment": None,
    "slice_processing_logs": [],
}


def init_slicing_workflow_state(state: MutableMapping[str, Any]) -> None:
    for key, value in SLICE_STATE_DEFAULTS.items():
        if key not in state:
            state[key] = _copy_default(value)


def reset_slicing_workflow_state(state: MutableMapping[str, Any]) -> None:
    for key, value in SLICE_STATE_DEFAULTS.items():
        state[key] = _copy_default(value)
    clear_preview_state(state)


def clear_preview_state(state: MutableMapping[str, Any]) -> None:
    for key in list(state.keys()):
        if str(key).startswith("slice_preview_"):
            del state[key]


def rebuild_flat_segments_from_export_segments(export_segments: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    flat_segments: list[dict[str, Any]] = []
    for idx, item in enumerate(export_segments or []):
        export_type = item.get("type", "主歌")
        source_segment = item.get("segment")
        flat_segments.append(
            {
                "id": f"seg_{idx:03d}",
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
                "songformer_label": item.get("songformer_label", ""),
                "current_label": export_type,
                "_initial_label": export_type,
                "is_highlight": bool(item.get("is_highlight", False)),
                "confidence": getattr(source_segment, "confidence", 0.9),
                "modified": False,
                "song_index": getattr(source_segment, "song_index", 0),
                "original_start": float(item.get("original_start", item.get("start", 0.0))),
                "original_end": float(item.get("original_end", item.get("end", 0.0))),
            }
        )
    return flat_segments


def apply_segment_edit(
    segments: list[dict[str, Any]],
    idx: int,
    new_label: str,
    new_start: float,
    new_end: float,
    new_sf_label: str | None = None,
) -> bool:
    if idx < 0 or idx >= len(segments):
        return False

    seg = segments[idx]
    has_changes = (
        new_label != seg.get("current_label")
        or abs(float(new_start) - float(seg.get("start", 0.0))) > 0.01
        or abs(float(new_end) - float(seg.get("end", 0.0))) > 0.01
        or (new_sf_label is not None and new_sf_label != seg.get("songformer_label", ""))
    )

    if not has_changes:
        return False

    seg["current_label"] = new_label
    seg["start"] = float(new_start)
    seg["end"] = float(new_end)
    if new_sf_label is not None:
        seg["songformer_label"] = new_sf_label
    seg["modified"] = True

    # Keep neighboring UI segments contiguous after manual boundary edits.
    if idx > 0:
        prev_seg = segments[idx - 1]
        if abs(float(new_start) - float(prev_seg.get("end", 0.0))) > 0.01:
            prev_seg["end"] = float(new_start)
            prev_seg["modified"] = True

    if idx < len(segments) - 1:
        next_seg = segments[idx + 1]
        if abs(float(new_end) - float(next_seg.get("start", 0.0))) > 0.01:
            next_seg["start"] = float(new_end)
            next_seg["modified"] = True

    return True


def reset_segment_edits(segments: list[dict[str, Any]]) -> None:
    for seg in segments:
        seg["current_label"] = seg.get("_initial_label", seg.get("current_label", "主歌"))
        seg["modified"] = False


def sync_ui_segments_to_export_segments(
    segments: Sequence[dict[str, Any]],
    export_segments: list[dict[str, Any]],
    songs_info: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply user edits from UI segments to the export payload.

    The processor exports from `export_segments`, while the editor mutates
    `slice_segments`. Keeping this mapping centralized prevents preview/export
    drift when labels, times, or song names are edited.
    """
    songs_by_index = {
        int(song.get("song_index", -1)): song
        for song in (songs_info or [])
        if song.get("song_index", -1) is not None
    }

    for idx, seg in enumerate(segments):
        if idx >= len(export_segments):
            break

        item = export_segments[idx]
        item["start"] = float(seg.get("start", item.get("start", 0.0)))
        item["end"] = float(seg.get("end", item.get("end", 0.0)))
        item["type"] = seg.get("current_label", item.get("type", "主歌"))
        item["songformer_label"] = seg.get("songformer_label", item.get("songformer_label", ""))

        song_info = songs_by_index.get(int(seg.get("song_index", 0)))
        song_obj = item.get("song")
        if song_info and song_obj:
            if hasattr(song_obj, "song_title"):
                song_obj.song_title = song_info.get("song_title", song_obj.song_title)
            if hasattr(song_obj, "song_artist"):
                song_obj.song_artist = song_info.get("song_artist", song_obj.song_artist)

    return export_segments


def _copy_default(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value
