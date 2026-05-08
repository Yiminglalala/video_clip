# -*- coding: utf-8 -*-
"""Post-processing helpers for export segment boundaries.

The functions in this module are intentionally pure-ish and operate on the
export-segment dictionaries built by ``LiveVideoProcessor``. Keeping these rules
outside the processor makes it easier to test overlap and duration behavior
without loading video, GPU models, or Streamlit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple


BOUNDARY_EPSILON = 0.05
DEFAULT_OVERLAP_TOLERANCE = 0.30

TYPE_PRIORITY = {
    "高光瞬间": 100,
    "合唱": 90,
    "副歌": 80,
    "主歌": 60,
    "乐器SOLO": 50,
    "讲话串场": 40,
}


def interval_overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def is_time_inside_sentence(time_point: float, sentences: List[Dict[str, Any]], epsilon: float = BOUNDARY_EPSILON) -> bool:
    for sent in sentences:
        try:
            start = float(sent["start"])
            end = float(sent["end"])
        except Exception:
            continue
        if start + epsilon < time_point < end - epsilon:
            return True
    return False


def build_global_asr_sentences(song: Any, cached_asr_results: Dict[int, dict]) -> List[Dict[str, Any]]:
    """Return cached ASR sentences on the absolute video timeline."""
    if not cached_asr_results:
        return []

    try:
        song_index = int(getattr(song, "song_index", 0))
        song_start_time = float(getattr(song, "start_time", 0.0))
    except Exception:
        return []

    asr_result = cached_asr_results.get(song_index) or {}
    sentences: List[Dict[str, Any]] = []
    for sent in asr_result.get("sentences", []) or []:
        try:
            start = float(sent.get("start", 0.0)) + song_start_time
            end = float(sent.get("end", 0.0)) + song_start_time
        except Exception:
            continue
        if end <= start:
            continue
        sentences.append({"start": start, "end": end, "text": str(sent.get("text", "") or "")})
    return sorted(sentences, key=lambda x: (float(x["start"]), float(x["end"])))


def _segment_priority(item: Dict[str, Any]) -> int:
    base = TYPE_PRIORITY.get(str(item.get("type", "")), 0)
    if item.get("is_highlight"):
        base += 20
    return base


def _candidate_score(
    prev_item: Dict[str, Any],
    next_item: Dict[str, Any],
    cut_point: float,
    reference_cut: float,
    min_duration: float,
    max_duration: float,
    base_score: float,
    reason: str,
) -> Optional[Tuple[int, float, float, float, str]]:
    prev_start = float(prev_item["start"])
    next_end = float(next_item["end"])
    min_gap = 0.10
    if cut_point <= prev_start + min_gap or cut_point >= next_end - min_gap:
        return None

    prev_duration = cut_point - prev_start
    next_duration = next_end - cut_point
    score = base_score + abs(cut_point - reference_cut) * 0.05

    duration_ok = prev_duration >= min_duration and next_duration >= min_duration
    max_ok = prev_duration <= max_duration * 1.10 and next_duration <= max_duration * 1.10
    hard_rank = 0 if duration_ok and max_ok else 1

    if prev_duration < min_duration:
        score += (min_duration - prev_duration) * 20.0
    if next_duration < min_duration:
        score += (min_duration - next_duration) * 20.0
    if prev_duration > max_duration * 1.10:
        score += (prev_duration - max_duration) * 4.0
    if next_duration > max_duration * 1.10:
        score += (next_duration - max_duration) * 4.0

    return hard_rank, score, abs(cut_point - reference_cut), cut_point, reason


def choose_sentence_aware_overlap_cut(
    prev_item: Dict[str, Any],
    next_item: Dict[str, Any],
    sentences: List[Dict[str, Any]],
    min_duration: float,
    max_duration: float,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Choose a non-overlap cut point while keeping subtitle sentences whole."""
    prev_start = float(prev_item["start"])
    prev_end = float(prev_item["end"])
    next_start = float(next_item["start"])
    next_end = float(next_item["end"])

    overlap_start = max(prev_start, next_start)
    overlap_end = min(prev_end, next_end)
    reference_cut = (overlap_start + overlap_end) / 2.0

    candidates: List[Tuple[int, float, float, float, str]] = []

    def add_candidate(cut_point: float, base_score: float, reason: str) -> None:
        candidate = _candidate_score(
            prev_item,
            next_item,
            cut_point,
            reference_cut,
            min_duration,
            max_duration,
            base_score,
            reason,
        )
        if candidate is not None:
            candidates.append(candidate)

    for left, right in zip(sentences, sentences[1:]):
        try:
            gap_start = float(left["end"])
            gap_end = float(right["start"])
        except Exception:
            continue
        if gap_end <= gap_start:
            continue
        if gap_end < overlap_start or gap_start > overlap_end:
            continue
        cut = (max(gap_start, overlap_start) + min(gap_end, overlap_end)) / 2.0
        add_candidate(cut, 0.0, "subtitle_gap")

    for sent in sentences:
        for key in ("start", "end"):
            try:
                boundary = float(sent[key])
            except Exception:
                continue
            if overlap_start <= boundary <= overlap_end:
                add_candidate(boundary, 1.0, f"sentence_{key}")

    for sent in sentences:
        try:
            sent_start = float(sent["start"])
            sent_end = float(sent["end"])
        except Exception:
            continue
        prev_cover = interval_overlap_seconds(sent_start, sent_end, prev_start, prev_end)
        next_cover = interval_overlap_seconds(sent_start, sent_end, next_start, next_end)
        if prev_cover <= 0.0 or next_cover <= 0.0:
            continue
        if prev_cover >= next_cover:
            add_candidate(sent_end, 2.0, "assign_sentence_to_prev")
        else:
            add_candidate(sent_start, 2.0, "assign_sentence_to_next")

    if not candidates or not sentences:
        add_candidate(reference_cut, 5.0, "overlap_midpoint")
    elif is_time_inside_sentence(reference_cut, sentences):
        add_candidate(reference_cut, 500.0, "midpoint_inside_sentence_fallback")
    else:
        add_candidate(reference_cut, 5.0, "overlap_midpoint")

    if not candidates:
        return reference_cut

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    hard_rank, best_score, _, best_cut, reason = candidates[0]
    if logger:
        logger.info(
            "[export overlap] choose cut %.2f reason=%s hard=%s score=%.2f prev=(%.2f-%.2f) next=(%.2f-%.2f)",
            best_cut,
            reason,
            hard_rank == 0,
            best_score,
            prev_start,
            prev_end,
            next_start,
            next_end,
        )
    return best_cut


def _merge_items(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(left)
    merged["start"] = min(float(left["start"]), float(right["start"]))
    merged["end"] = max(float(left["end"]), float(right["end"]))
    merged["is_highlight"] = bool(left.get("is_highlight") or right.get("is_highlight"))
    if _segment_priority(right) > _segment_priority(left):
        merged["type"] = right.get("type", merged.get("type"))
        merged["songformer_label"] = right.get("songformer_label", merged.get("songformer_label", ""))
    return merged


def _same_song(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_song = left.get("song")
    right_song = right.get("song")
    return int(getattr(left_song, "song_index", -1)) == int(getattr(right_song, "song_index", -2))


def _repair_short_segments(
    segments: List[Dict[str, Any]],
    min_duration: float,
    max_duration: float,
    overlap_tolerance: float,
    logger: Optional[logging.Logger],
) -> List[Dict[str, Any]]:
    if len(segments) <= 1:
        if segments and float(segments[0]["end"]) - float(segments[0]["start"]) < min_duration and logger:
            logger.warning(
                "[export duration] keeping only segment below min duration because no neighbor exists: %.2fs < %.2fs",
                float(segments[0]["end"]) - float(segments[0]["start"]),
                min_duration,
            )
        return segments

    items = sorted([dict(seg) for seg in segments], key=lambda x: (float(x["start"]), float(x["end"])))
    i = 0
    while i < len(items):
        item = items[i]
        duration = float(item["end"]) - float(item["start"])
        if duration >= min_duration - BOUNDARY_EPSILON:
            i += 1
            continue

        missing = min_duration - duration
        fixed = False

        if i + 1 < len(items):
            next_item = items[i + 1]
            gap = float(next_item["start"]) - float(item["end"])
            next_duration = float(next_item["end"]) - float(next_item["start"])
            if gap <= overlap_tolerance and next_duration - missing >= min_duration:
                item["end"] = round(float(item["end"]) + missing, 3)
                next_item["start"] = item["end"]
                fixed = True

        if not fixed and i > 0:
            prev_item = items[i - 1]
            gap = float(item["start"]) - float(prev_item["end"])
            prev_duration = float(prev_item["end"]) - float(prev_item["start"])
            if gap <= overlap_tolerance and prev_duration - missing >= min_duration:
                item["start"] = round(float(item["start"]) - missing, 3)
                prev_item["end"] = item["start"]
                fixed = True

        if fixed:
            if logger:
                logger.info("[export duration] repaired short segment to %.2fs", float(item["end"]) - float(item["start"]))
            i = max(0, i - 1)
            continue

        merge_target = None
        if i > 0 and items[i - 1].get("type") == item.get("type") and _same_song(items[i - 1], item):
            combined = float(item["end"]) - float(items[i - 1]["start"])
            if combined <= max_duration + BOUNDARY_EPSILON:
                merge_target = i - 1
        if merge_target is None and i + 1 < len(items) and items[i + 1].get("type") == item.get("type") and _same_song(item, items[i + 1]):
            combined = float(items[i + 1]["end"]) - float(item["start"])
            if combined <= max_duration + BOUNDARY_EPSILON:
                merge_target = i + 1

        if merge_target is not None:
            target = items[merge_target]
            merged = _merge_items(target, item) if merge_target < i else _merge_items(item, target)
            keep_idx = min(i, merge_target)
            drop_idx = max(i, merge_target)
            items[keep_idx] = merged
            del items[drop_idx]
            if logger:
                logger.info("[export duration] merged short same-type segment into %.2fs", float(merged["end"]) - float(merged["start"]))
            i = max(0, keep_idx - 1)
            continue

        if logger:
            logger.warning(
                "[export duration] dropped short segment %.2fs < %.2fs: %.2f-%.2f type=%s",
                duration,
                min_duration,
                float(item["start"]),
                float(item["end"]),
                item.get("type"),
            )
        del items[i]
        i = max(0, i - 1)

    return items


def _remaining_overlaps(segments: Iterable[Dict[str, Any]], overlap_tolerance: float) -> List[Tuple[float, float, float, float, float]]:
    items = list(segments)
    overlaps = []
    for left, right in zip(items, items[1:]):
        overlap = float(left["end"]) - float(right["start"])
        if overlap > overlap_tolerance:
            overlaps.append((float(left["start"]), float(left["end"]), float(right["start"]), float(right["end"]), overlap))
    return overlaps


def normalize_export_segment_overlaps(
    segments: List[Dict[str, Any]],
    min_duration: float,
    max_duration: float,
    cached_asr_results: Optional[Dict[int, dict]] = None,
    overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Make adjacent export segments non-overlapping and min-duration safe."""
    if len(segments) <= 1:
        return segments

    normalized = [dict(seg) for seg in sorted(segments, key=lambda x: (float(x["start"]), float(x["end"])))]
    cached_asr_results = cached_asr_results or {}
    sentence_cache: Dict[int, List[Dict[str, Any]]] = {}
    fixed_count = 0

    for idx in range(len(normalized) - 1):
        prev_item = normalized[idx]
        next_item = normalized[idx + 1]
        prev_end = float(prev_item["end"])
        next_start = float(next_item["start"])
        overlap = prev_end - next_start
        if overlap <= overlap_tolerance:
            if overlap > 0:
                prev_item["end"] = next_start
                fixed_count += 1
            continue

        prev_song = prev_item.get("song")
        next_song = next_item.get("song")
        prev_song_index = int(getattr(prev_song, "song_index", -1))
        next_song_index = int(getattr(next_song, "song_index", -2))
        if prev_song_index != next_song_index:
            cut_point = next_start
        else:
            if prev_song_index not in sentence_cache:
                sentence_cache[prev_song_index] = build_global_asr_sentences(prev_song, cached_asr_results)
            cut_point = choose_sentence_aware_overlap_cut(
                prev_item,
                next_item,
                sentence_cache.get(prev_song_index, []),
                min_duration,
                max_duration,
                logger=logger,
            )

        old_prev_end = prev_end
        old_next_start = next_start
        prev_item["end"] = round(float(cut_point), 3)
        next_item["start"] = round(float(cut_point), 3)
        fixed_count += 1
        if logger:
            logger.info(
                "[export overlap] fixed overlap %.2fs: prev_end %.2f->%.2f next_start %.2f->%.2f",
                overlap,
                old_prev_end,
                float(prev_item["end"]),
                old_next_start,
                float(next_item["start"]),
            )

    final_segments = [
        seg
        for seg in normalized
        if float(seg["end"]) > float(seg["start"]) + BOUNDARY_EPSILON
    ]
    final_segments = _repair_short_segments(final_segments, min_duration, max_duration, overlap_tolerance, logger)

    remaining_overlaps = _remaining_overlaps(final_segments, overlap_tolerance)
    if fixed_count and logger:
        logger.info("[export overlap] normalized %s adjacent overlaps", fixed_count)
    if remaining_overlaps and logger:
        logger.warning("[export overlap] remaining overlaps after normalization: %s", remaining_overlaps[:5])

    return final_segments
