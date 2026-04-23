#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subtitle-aware alignment for export segments."""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BOUNDARY_EPSILON = 0.05
_SENTENCE_GUARD_SHIFT = 4.0
_OVERLAP_SOFT_MIN_DURATION = 4.0
_SAFETY_MARGIN = 0.15  # 安全边距：在字幕边界前切断


def _build_global_sentences(
    song_index: int,
    cached_asr_results: Dict[int, dict],
    song_start_time: float = 0.0,
) -> List[dict]:
    if not cached_asr_results or song_index not in cached_asr_results:
        return []

    asr_result = cached_asr_results[song_index] or {}
    sentences = asr_result.get("sentences", []) or []
    result: List[dict] = []
    for sent in sentences:
        try:
            start = float(sent.get("start", 0.0)) + song_start_time
            end = float(sent.get("end", 0.0)) + song_start_time
        except Exception:
            continue
        if end <= start:
            continue
        result.append(
            {
                "start": start,
                "end": end,
                "text": str(sent.get("text", "") or ""),
            }
        )
    return sorted(result, key=lambda x: (x["start"], x["end"]))


def _find_closest_subtitle_boundary(
    time_point: float,
    sentences: List[dict],
    boundary_type: str = "end",
    max_shift: float = 2.0,
    force_align: bool = False,
) -> float:
    if not sentences:
        return time_point

    boundaries: List[float] = []
    for sent in sentences:
        if boundary_type == "start":
            boundaries.append(float(sent["start"]))
        elif boundary_type == "end":
            boundaries.append(float(sent["end"]))

    if not boundaries:
        return time_point

    closest = min(boundaries, key=lambda boundary: abs(boundary - time_point))
    min_diff = abs(closest - time_point)
    if min_diff <= max_shift or force_align:
        logger.debug(
            "[字幕对齐] %s 边界吸附: %.2f -> %.2f (shift=%.2fs)",
            boundary_type,
            time_point,
            closest,
            min_diff,
        )
        return closest
    return time_point


def _find_sentence_covering_time(
    time_point: float,
    sentences: List[dict],
    epsilon: float = _BOUNDARY_EPSILON,
) -> Optional[dict]:
    for sent in sentences:
        start = float(sent["start"])
        end = float(sent["end"])
        if start + epsilon < time_point < end - epsilon:
            return sent
    return None


def _expand_to_sentence_boundary(
    time_point: float,
    sentences: List[dict],
    boundary_type: str,
    max_shift: float,
    force_align: bool = False,
) -> float:
    sentence = _find_sentence_covering_time(time_point, sentences)
    
    # ── 智能策略：对于结束边界，优先选择在句子之前切断 ──
    if boundary_type == "end":
        # 寻找最近的、在时间点之前的句子结束点
        earlier_ends = [
            float(sent["end"]) for sent in sentences if float(sent["end"]) <= time_point
        ]
        if earlier_ends:
            # 找最近的更早的句子结束点
            closest_earlier_end = max(earlier_ends)
            shift = time_point - closest_earlier_end
            if shift <= max_shift + _SAFETY_MARGIN:
                # 在句子结束点前加上安全边距
                safe_candidate = closest_earlier_end - _SAFETY_MARGIN
                logger.debug(
                    "[字幕对齐] 结束边界智能策略：在句前切断: %.2f -> %.2f (shift=%.2fs, text=%s)",
                    time_point,
                    safe_candidate,
                    shift,
                    str(sentence.get("text", "") if sentence else "")[:40],
                )
                return safe_candidate
    
    if sentence is None:
        return _find_closest_subtitle_boundary(
            time_point,
            sentences,
            boundary_type=boundary_type,
            max_shift=max_shift,
            force_align=force_align,
        )

    candidate = float(sentence["start"] if boundary_type == "start" else sentence["end"])
    shift = abs(candidate - time_point)
    guard_shift = max(max_shift, _SENTENCE_GUARD_SHIFT)
    
    # ── 对于结束边界，使用保守策略
    if boundary_type == "end":
        # 如果要扩展到句子结束，加上安全边距
        if candidate > time_point:
            candidate = candidate - _SAFETY_MARGIN
    
    if shift <= guard_shift or force_align:
        logger.debug(
            "[字幕对齐] %s 命中句内，扩展到整句边界: %.2f -> %.2f (shift=%.2fs, text=%s)",
            boundary_type,
            time_point,
            candidate,
            shift,
            str(sentence.get("text", ""))[:40],
        )
        return candidate

    return _find_closest_subtitle_boundary(
        time_point,
        sentences,
        boundary_type=boundary_type,
        max_shift=max_shift,
        force_align=force_align,
    )


def _cover_overlapping_sentences(
    seg_start: float,
    seg_end: float,
    sentences: List[dict],
    max_shift: float,
    force_align: bool = False,
) -> Tuple[float, float]:
    if not sentences:
        return seg_start, seg_end

    overlap = [
        sent
        for sent in sentences
        if float(sent["start"]) <= seg_end and float(sent["end"]) >= seg_start
    ]
    if not overlap:
        return seg_start, seg_end

    guard_shift = max(max_shift, _SENTENCE_GUARD_SHIFT)
    first = overlap[0]
    last = overlap[-1]

    fixed_start = seg_start
    fixed_end = seg_end
    if float(first["start"]) < fixed_start:
        shift = fixed_start - float(first["start"])
        if shift <= guard_shift or force_align:
            fixed_start = float(first["start"])
    if float(last["end"]) > fixed_end:
        shift = float(last["end"]) - fixed_end
        if shift <= guard_shift or force_align:
            fixed_end = float(last["end"])

    return fixed_start, fixed_end


def _get_segment_song_index(item: dict) -> int:
    return int(getattr(item.get("segment"), "song_index", 0))


def _sentence_intersects_window(sent: dict, start: float, end: float) -> bool:
    return float(sent["start"]) < end - _BOUNDARY_EPSILON and float(sent["end"]) > start + _BOUNDARY_EPSILON


def _score_overlap_cut(prev_item: dict, next_item: dict, cut_point: float) -> float:
    prev_duration = cut_point - float(prev_item["start"])
    next_duration = float(next_item["end"]) - cut_point
    if prev_duration <= _BOUNDARY_EPSILON or next_duration <= _BOUNDARY_EPSILON:
        return float("inf")

    prev_original_end = float(prev_item.get("original_end", prev_item["end"]))
    next_original_start = float(next_item.get("original_start", next_item["start"]))
    score = abs(prev_original_end - cut_point) + abs(next_original_start - cut_point)

    if prev_duration < _OVERLAP_SOFT_MIN_DURATION:
        score += (_OVERLAP_SOFT_MIN_DURATION - prev_duration) * 100.0
    if next_duration < _OVERLAP_SOFT_MIN_DURATION:
        score += (_OVERLAP_SOFT_MIN_DURATION - next_duration) * 100.0
    return score


def _choose_overlap_cut(
    prev_item: dict,
    next_item: dict,
    sentences: List[dict],
) -> Optional[float]:
    overlap_start = float(next_item["start"])
    overlap_end = float(prev_item["end"])
    if overlap_end <= overlap_start + _BOUNDARY_EPSILON:
        return None

    shared_sentences = [
        sent
        for sent in sentences
        if _sentence_intersects_window(sent, float(prev_item["start"]), float(prev_item["end"]))
        and _sentence_intersects_window(sent, float(next_item["start"]), float(next_item["end"]))
    ]

    candidates: List[float] = []
    if shared_sentences:
        candidates.extend(
            [
                float(shared_sentences[0]["start"]),
                float(shared_sentences[-1]["end"]),
            ]
        )
    else:
        for sent in sentences:
            for boundary in (float(sent["start"]), float(sent["end"])):
                if overlap_start - _BOUNDARY_EPSILON <= boundary <= overlap_end + _BOUNDARY_EPSILON:
                    candidates.append(boundary)

    reference_cut = (
        float(prev_item.get("original_end", prev_item["end"]))
        + float(next_item.get("original_start", next_item["start"]))
    ) / 2.0
    reference_cut = max(overlap_start, min(overlap_end, reference_cut))

    unique_candidates = sorted(set(round(c, 6) for c in candidates))
    if not unique_candidates:
        unique_candidates = [round(reference_cut, 6)]

    best_cut: Optional[float] = None
    best_score = float("inf")
    for cut_point in unique_candidates:
        score = _score_overlap_cut(prev_item, next_item, cut_point)
        if score < best_score:
            best_score = score
            best_cut = cut_point
    return best_cut


def _deduplicate_adjacent_overlaps(
    aligned_segments: List[dict],
    sentence_lookup: Dict[int, List[dict]],
) -> List[dict]:
    if len(aligned_segments) <= 1:
        return aligned_segments

    song_groups: Dict[int, List[dict]] = {}
    for item in aligned_segments:
        song_groups.setdefault(_get_segment_song_index(item), []).append(item)

    fixed_pairs = 0
    for song_index, items in song_groups.items():
        items.sort(key=lambda x: (float(x["start"]), float(x["end"])))
        sentences = sentence_lookup.get(song_index, [])
        for idx in range(len(items) - 1):
            prev_item = items[idx]
            next_item = items[idx + 1]
            prev_end = float(prev_item["end"])
            next_start = float(next_item["start"])
            if next_start >= prev_end - _BOUNDARY_EPSILON:
                continue

            cut_point = _choose_overlap_cut(prev_item, next_item, sentences)
            if cut_point is None:
                continue

            old_prev_end = prev_end
            old_next_start = next_start
            prev_item["end"] = cut_point
            next_item["start"] = cut_point
            fixed_pairs += 1
            logger.info(
                "[subtitle alignment] de-overlap song=%s prev_end %.2f->%.2f next_start %.2f->%.2f cut=%.2f",
                song_index,
                old_prev_end,
                float(prev_item["end"]),
                old_next_start,
                float(next_item["start"]),
                cut_point,
            )

    if fixed_pairs:
        logger.info("[subtitle alignment] fixed %s adjacent overlaps", fixed_pairs)
    return aligned_segments


def align_segment_to_subtitles(
    original_start: float,
    original_end: float,
    song_index: int,
    cached_asr_results: Dict[int, dict],
    song_start_time: float = 0.0,
    max_shift: float = 2.0,
    force_align: bool = False,
) -> Tuple[float, float]:
    if not cached_asr_results or song_index not in cached_asr_results:
        logger.warning("[字幕对齐] 歌曲%s 无 ASR 结果，跳过对齐", song_index)
        return original_start, original_end

    sentences = _build_global_sentences(song_index, cached_asr_results, song_start_time)
    if not sentences:
        logger.warning("[字幕对齐] 歌曲%s ASR 无有效句子，跳过对齐", song_index)
        return original_start, original_end

    aligned_start = _expand_to_sentence_boundary(
        original_start,
        sentences,
        boundary_type="start",
        max_shift=max_shift,
        force_align=force_align,
    )
    aligned_end = _expand_to_sentence_boundary(
        original_end,
        sentences,
        boundary_type="end",
        max_shift=max_shift,
        force_align=force_align,
    )
    aligned_start, aligned_end = _cover_overlapping_sentences(
        aligned_start,
        aligned_end,
        sentences,
        max_shift=max_shift,
        force_align=force_align,
    )

    if aligned_end <= aligned_start:
        logger.warning(
            "[字幕对齐] 结果异常，回退原始边界: (%.2f-%.2f) -> (%.2f-%.2f)",
            original_start,
            original_end,
            aligned_start,
            aligned_end,
        )
        return original_start, original_end

    logger.info(
        "[字幕对齐] 歌曲%s: (%.2f-%.2f) -> (%.2f-%.2f)",
        song_index,
        original_start,
        original_end,
        aligned_start,
        aligned_end,
    )
    return aligned_start, aligned_end


def align_all_segments_to_subtitles(
    export_segments: List[dict],
    cached_asr_results: Dict[int, dict],
    songs: List,
    max_shift: float = 2.0,
    force_align: bool = False,
) -> List[dict]:
    if not cached_asr_results:
        logger.warning("[字幕对齐] 无 ASR 缓存，跳过所有片段对齐")
        return export_segments

    song_start_lookup = {
        int(getattr(song, "song_index", -1)): float(getattr(song, "start_time", 0.0))
        for song in songs
    }

    sentence_lookup = {
        song_index: _build_global_sentences(song_index, cached_asr_results, song_start_time)
        for song_index, song_start_time in song_start_lookup.items()
    }

    aligned_segments: List[dict] = []
    for item in export_segments:
        song_index = _get_segment_song_index(item)
        song_start_time = song_start_lookup.get(song_index, 0.0)
        original_start = float(item["start"])
        original_end = float(item["end"])
        aligned_start, aligned_end = align_segment_to_subtitles(
            original_start,
            original_end,
            song_index,
            cached_asr_results,
            song_start_time=song_start_time,
            max_shift=max_shift,
            force_align=force_align,
        )
        aligned_item = dict(item)
        aligned_item["start"] = aligned_start
        aligned_item["end"] = aligned_end
        aligned_item["original_start"] = original_start
        aligned_item["original_end"] = original_end
        aligned_segments.append(aligned_item)

    aligned_segments = _deduplicate_adjacent_overlaps(aligned_segments, sentence_lookup)

    logger.info("[字幕对齐] 完成: 共对齐 %s 个片段", len(aligned_segments))
    return aligned_segments


def get_segment_subtitles(
    segment_start: float,
    segment_end: float,
    song_index: int,
    cached_asr_results: Dict[int, dict],
    song_start_time: float = 0.0,
) -> List[dict]:
    sentences = _build_global_sentences(song_index, cached_asr_results, song_start_time)
    if not sentences:
        return []

    segment_subtitles: List[dict] = []
    for sent in sentences:
        start = float(sent["start"])
        end = float(sent["end"])
        if start < segment_end - _BOUNDARY_EPSILON and end > segment_start + _BOUNDARY_EPSILON:
            segment_subtitles.append(
                {
                    "start": start,
                    "end": end,
                    "text": sent.get("text", ""),
                    "is_truncated_start": start < segment_start - _BOUNDARY_EPSILON,
                    "is_truncated_end": end > segment_end + _BOUNDARY_EPSILON,
                }
            )
    return segment_subtitles


def check_subtitle_cuts(
    export_segments: List[dict],
    cached_asr_results: Dict[int, dict],
    songs: List,
) -> List[dict]:
    cut_detected: List[dict] = []
    song_start_lookup = {
        int(getattr(song, "song_index", -1)): float(getattr(song, "start_time", 0.0))
        for song in songs
    }

    for idx, item in enumerate(export_segments):
        song_index = int(getattr(item.get("segment"), "song_index", 0))
        seg_start = float(item["start"])
        seg_end = float(item["end"])
        subtitles = get_segment_subtitles(
            seg_start,
            seg_end,
            song_index,
            cached_asr_results,
            song_start_lookup.get(song_index, 0.0),
        )
        for sub in subtitles:
            if not sub.get("is_truncated_start") and not sub.get("is_truncated_end"):
                continue
            cut_detected.append(
                {
                    "segment_idx": idx,
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "subtitle_text": sub.get("text", ""),
                    "subtitle_start": sub.get("start"),
                    "subtitle_end": sub.get("end"),
                    "cut_type": "start" if sub.get("is_truncated_start") else "end",
                    "song_index": song_index,
                }
            )

    logger.info("[字幕检查] 检测到 %s 个被切断的歌词", len(cut_detected))
    return cut_detected
