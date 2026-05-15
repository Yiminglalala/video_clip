"""Preview generation service for the slicing page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.output_spec import OutputResolutionSpec


@dataclass(frozen=True)
class SegmentPreviewResult:
    success: bool
    path: str | None = None
    message: str = ""
    subtitle_error: str = ""


def build_preview_cache_key(selected_idx: int, segment: dict[str, Any], output_spec: OutputResolutionSpec) -> str:
    return (
        f"preview_v3_{selected_idx}_{float(segment['start']):.2f}_{float(segment['end']):.2f}_"
        f"{output_spec.width}x{output_spec.height}_fs90_mv240"
    )


def find_song_start_time(songs: Sequence[Any], song_index: int) -> float:
    for song in songs or []:
        if getattr(song, "song_index", -1) == song_index:
            return float(getattr(song, "start_time", 0.0))
    return 0.0


def generate_segment_preview(
    *,
    video_source: str,
    segment: dict[str, Any],
    selected_idx: int,
    output_spec: OutputResolutionSpec,
    temp_manager: Any,
    cached_asr_results: dict[Any, Any] | None = None,
    songs: Sequence[Any] | None = None,
    enable_subtitle: bool = False,
) -> SegmentPreviewResult:
    from src.ffmpeg_processor import FFmpegProcessor
    from src.processor import LiveVideoProcessor, ProcessingConfig

    safe_start = float(segment["start"])
    safe_end = float(segment["end"])
    song_index = int(segment.get("song_index", 0))

    ffmpeg = FFmpegProcessor()
    temp_no_subtitle = temp_manager.get_preview_path(f"temp_preview_nosub_{selected_idx}.mp4")
    cut_result = ffmpeg.cut_video(
        video_source,
        safe_start,
        safe_end,
        temp_no_subtitle,
        mode="accurate",
        output_spec=output_spec,
        safety_margin=0.0,
    )
    if not cut_result.success:
        return SegmentPreviewResult(False, message=cut_result.error_message or "preview cut failed")

    temp_final = temp_no_subtitle
    subtitle_error = ""
    cached_asr_results = cached_asr_results or {}
    if enable_subtitle:
        cached = cached_asr_results.get(song_index)
        if cached and not cached.get("error"):
            song_start_time = find_song_start_time(songs or [], song_index)
            relative_start = safe_start - song_start_time
            relative_end = safe_end - song_start_time

            temp_subtitled = temp_manager.get_preview_path(f"temp_preview_sub_{selected_idx}.mp4")
            config = ProcessingConfig(
                output_dir=temp_manager.get_cache_path(""),
                enable_subtitle=True,
            )
            processor = LiveVideoProcessor(config)
            ok, result_path = processor._generate_subtitles_from_cached_asr(
                temp_no_subtitle,
                cached,
                relative_start,
                relative_end,
                temp_subtitled,
                output_spec,
            )
            if ok:
                temp_final = result_path
            else:
                subtitle_error = str(result_path)

    return SegmentPreviewResult(True, path=temp_final, subtitle_error=subtitle_error)
