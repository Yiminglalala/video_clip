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
ADJACENT_BOUNDARY_TOLERANCE = 0.50
SENTENCE_END_NEAR_TOLERANCE = 0.12
SENTENCE_TAIL_PADDING = 0.35

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


def build_global_asr_words(song: Any, cached_asr_results: Dict[int, dict]) -> List[Dict[str, Any]]:
    """Return cached ASR words on the absolute video timeline."""
    if not cached_asr_results:
        return []

    try:
        song_index = int(getattr(song, "song_index", 0))
        song_start_time = float(getattr(song, "start_time", 0.0))
    except Exception:
        return []

    asr_result = cached_asr_results.get(song_index) or {}
    words: List[Dict[str, Any]] = []
    for word in asr_result.get("words", []) or []:
        try:
            start = float(word.get("start", 0.0)) + song_start_time
            end = float(word.get("end", 0.0)) + song_start_time
        except Exception:
            continue
        if end <= start:
            continue
        text = str(word.get("word") or word.get("text") or "")
        words.append({"start": start, "end": end, "word": text})
    return sorted(words, key=lambda x: (float(x["start"]), float(x["end"])))


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


def _find_sentence_containing_boundary(
    boundary: float,
    sentences: List[Dict[str, Any]],
    epsilon: float = BOUNDARY_EPSILON,
) -> Optional[Dict[str, Any]]:
    for sent in sentences:
        try:
            start = float(sent["start"])
            end = float(sent["end"])
        except Exception:
            continue
        if start + epsilon < boundary < end - epsilon:
            return sent
    return None


def _find_sentence_ending_near_boundary(
    boundary: float,
    sentences: List[Dict[str, Any]],
    tolerance: float = SENTENCE_END_NEAR_TOLERANCE,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_delta = float("inf")
    for sent in sentences:
        try:
            end = float(sent["end"])
        except Exception:
            continue
        delta = boundary - end
        if 0.0 <= delta <= tolerance and delta < best_delta:
            best = sent
            best_delta = delta
    return best


def _next_sentence_start_after(sent_end: float, sentences: List[Dict[str, Any]]) -> Optional[float]:
    starts: List[float] = []
    for sent in sentences:
        try:
            start = float(sent["start"])
        except Exception:
            continue
        if start > sent_end + BOUNDARY_EPSILON:
            starts.append(start)
    return min(starts) if starts else None


def _words_overlapping_sentence(
    sentence: Dict[str, Any],
    words: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not words:
        return []
    try:
        sent_start = float(sentence["start"])
        sent_end = float(sentence["end"])
    except Exception:
        return []

    matched: List[Dict[str, Any]] = []
    for word in words:
        try:
            word_start = float(word["start"])
            word_end = float(word["end"])
        except Exception:
            continue
        if interval_overlap_seconds(sent_start, sent_end, word_start, word_end) > 0.0:
            matched.append(word)
    return matched


def _word_unit_count(word: Dict[str, Any]) -> int:
    text = str(word.get("word") or word.get("text") or "")
    compact = "".join(ch for ch in text if not ch.isspace())
    return max(1, len(compact))


def _sentence_word_progress(
    sentence: Dict[str, Any],
    boundary: float,
    words: Optional[List[Dict[str, Any]]],
) -> Optional[Tuple[int, int, float]]:
    sentence_words = _words_overlapping_sentence(sentence, words)
    if not sentence_words:
        return None

    prev_count = 0
    total = 0
    for word in sentence_words:
        try:
            word_start = float(word["start"])
            word_end = float(word["end"])
        except Exception:
            continue

        units = _word_unit_count(word)
        total += units
        if boundary >= word_end:
            prev_count += units
        elif boundary > word_start:
            progress = (boundary - word_start) / max(BOUNDARY_EPSILON, word_end - word_start)
            prev_count += max(0, min(units, int(round(units * progress))))

    if total <= 0:
        return None
    return prev_count, total, prev_count / total


def choose_sentence_complete_adjacent_cut(
    prev_item: Dict[str, Any],
    next_item: Dict[str, Any],
    sentence: Dict[str, Any],
    boundary: float,
    min_duration: float,
    max_duration: float,
    words: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[float]:
    """Move an adjacent boundary out of a subtitle sentence.

    Overlap normalization only handles segments that overlap. This helper fixes
    the separate case where two adjacent segments share a boundary that falls
    inside one lyric sentence, which otherwise creates a visibly/audibly cut
    final line.
    """
    try:
        sent_start = float(sentence["start"])
        sent_end = float(sentence["end"])
    except Exception:
        return None

    if sent_end <= sent_start:
        return None

    sent_len = sent_end - sent_start
    prev_cover = max(0.0, boundary - sent_start)
    tail_remaining = max(0.0, sent_end - boundary)
    word_progress = _sentence_word_progress(sentence, boundary, words)
    if word_progress is not None:
        prev_word_count, total_word_count, word_ratio = word_progress
        # Word-level ownership is stricter than the old time-only heuristic:
        # a line that has only started by a few words usually belongs with the
        # next clip, not with the previous clip.
        previous_owns_sentence = (
            word_ratio >= 0.60 and (prev_word_count > 3 or tail_remaining <= 0.45)
        ) or (
            tail_remaining <= 0.45 and word_ratio >= 0.50
        )
        progress_reason = f"words={prev_word_count}/{total_word_count}"
    else:
        prev_ratio = prev_cover / sent_len
        previous_owns_sentence = prev_ratio >= 0.55 or tail_remaining <= 0.45
        progress_reason = f"time_ratio={prev_ratio:.2f}"

    candidates: List[Tuple[int, float, float, float, str]] = []

    def add_candidate(cut_point: float, base_score: float, reason: str) -> None:
        candidate = _candidate_score(
            prev_item,
            next_item,
            cut_point,
            boundary,
            min_duration,
            max_duration,
            base_score,
            reason,
        )
        if candidate is not None:
            candidates.append(candidate)

    add_candidate(
        sent_end,
        0.0 if previous_owns_sentence else 3.0,
        f"complete_sentence_to_prev_{progress_reason}",
    )
    add_candidate(
        sent_start,
        0.0 if not previous_owns_sentence else 3.5,
        f"complete_sentence_to_next_{progress_reason}",
    )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    hard_rank, best_score, _, best_cut, reason = candidates[0]
    if abs(best_cut - boundary) <= BOUNDARY_EPSILON:
        return None
    if logger:
        logger.info(
            "[export sentence-boundary] move cut %.2f->%.2f reason=%s hard=%s score=%.2f sentence=(%.2f-%.2f)",
            boundary,
            best_cut,
            reason,
            hard_rank == 0,
            best_score,
            sent_start,
            sent_end,
        )
    return best_cut


def choose_sentence_tail_padding_cut(
    prev_item: Dict[str, Any],
    next_item: Dict[str, Any],
    sentence: Dict[str, Any],
    next_sentence_start: Optional[float],
    boundary: float,
    min_duration: float,
    max_duration: float,
    logger: Optional[logging.Logger] = None,
) -> Optional[float]:
    """Add a tiny tail pad when a cut is exactly at an ASR sentence end."""
    try:
        sent_end = float(sentence["end"])
    except Exception:
        return None

    padded_cut = sent_end + SENTENCE_TAIL_PADDING
    if next_sentence_start is not None:
        padded_cut = min(padded_cut, max(sent_end, next_sentence_start - BOUNDARY_EPSILON))
    if padded_cut <= boundary + BOUNDARY_EPSILON:
        return None

    candidate = _candidate_score(
        prev_item,
        next_item,
        padded_cut,
        boundary,
        min_duration,
        max_duration,
        0.5,
        "sentence_end_tail_pad",
    )
    if candidate is None:
        return None

    hard_rank, best_score, _, best_cut, reason = candidate
    if logger:
        logger.info(
            "[export sentence-boundary] move cut %.2f->%.2f reason=%s hard=%s score=%.2f sentence_end=%.2f",
            boundary,
            best_cut,
            reason,
            hard_rank == 0,
            best_score,
            sent_end,
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


def _song_index(item: Dict[str, Any]) -> int:
    return int(getattr(item.get("song"), "song_index", -1))


def _get_item_sentences(
    item: Dict[str, Any],
    sentence_cache: Dict[int, List[Dict[str, Any]]],
    cached_asr_results: Dict[int, dict],
) -> List[Dict[str, Any]]:
    song_idx = _song_index(item)
    if song_idx < 0:
        return []
    if song_idx not in sentence_cache:
        sentence_cache[song_idx] = build_global_asr_sentences(item.get("song"), cached_asr_results)
    return sentence_cache.get(song_idx, [])


def _sentence_overlap_seconds(
    start: float,
    end: float,
    sentences: List[Dict[str, Any]],
) -> float:
    total = 0.0
    for sentence in sentences:
        try:
            sent_start = float(sentence["start"])
            sent_end = float(sentence["end"])
        except Exception:
            continue
        total += interval_overlap_seconds(start, end, sent_start, sent_end)
    return total


def _segment_asr_coverage_seconds(
    item: Dict[str, Any],
    sentence_cache: Dict[int, List[Dict[str, Any]]],
    cached_asr_results: Dict[int, dict],
) -> float:
    sentences = _get_item_sentences(item, sentence_cache, cached_asr_results)
    return _sentence_overlap_seconds(float(item["start"]), float(item["end"]), sentences)


def _split_range_for_gap(start: float, end: float, max_duration: float) -> List[Tuple[float, float]]:
    duration = max(0.0, end - start)
    if duration <= BOUNDARY_EPSILON:
        return []
    if duration <= max_duration + BOUNDARY_EPSILON:
        return [(round(start, 3), round(end, 3))]

    part_count = max(1, int((duration + max_duration - BOUNDARY_EPSILON) // max_duration))
    if start + part_count * max_duration < end - BOUNDARY_EPSILON:
        part_count += 1
    chunk = duration / part_count
    parts: List[Tuple[float, float]] = []
    cursor = start
    for idx in range(part_count):
        next_cursor = end if idx == part_count - 1 else start + chunk * (idx + 1)
        parts.append((round(cursor, 3), round(next_cursor, 3)))
        cursor = next_cursor
    return parts


def _repair_short_segments(
    segments: List[Dict[str, Any]],
    min_duration: float,
    max_duration: float,
    overlap_tolerance: float,
    logger: Optional[logging.Logger],
    sentence_cache: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    cached_asr_results: Optional[Dict[int, dict]] = None,
) -> List[Dict[str, Any]]:
    if len(segments) <= 1:
        if segments and float(segments[0]["end"]) - float(segments[0]["start"]) < min_duration and logger:
            logger.warning(
                "[export duration] keeping only segment below min duration because no neighbor exists: %.2fs < %.2fs",
                float(segments[0]["end"]) - float(segments[0]["start"]),
                min_duration,
            )
        return segments

    sentence_cache = sentence_cache if sentence_cache is not None else {}
    cached_asr_results = cached_asr_results or {}
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

        # Legacy conservative repair: only borrow exact missing duration if the
        # neighbor can still satisfy the minimum duration immediately.
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

        asr_coverage = _segment_asr_coverage_seconds(item, sentence_cache, cached_asr_results)
        if asr_coverage >= 0.30:
            if logger:
                logger.warning(
                    "[export duration] keeping lyrics-covered short segment %.2fs < %.2fs: %.2f-%.2f type=%s asr=%.2fs",
                    duration,
                    min_duration,
                    float(item["start"]),
                    float(item["end"]),
                    item.get("type"),
                    asr_coverage,
                )
            i += 1
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


def _fill_asr_covered_gaps(
    segments: List[Dict[str, Any]],
    sentence_cache: Dict[int, List[Dict[str, Any]]],
    cached_asr_results: Dict[int, dict],
    min_duration: float,
    max_duration: float,
    logger: Optional[logging.Logger],
) -> Tuple[List[Dict[str, Any]], int]:
    if len(segments) <= 1:
        return segments, 0

    items = sorted([dict(seg) for seg in segments], key=lambda x: (float(x["start"]), float(x["end"])))
    filled: List[Dict[str, Any]] = []
    fill_count = 0

    for idx, item in enumerate(items):
        filled.append(item)
        if idx >= len(items) - 1:
            continue

        next_item = items[idx + 1]
        gap_start = float(item["end"])
        gap_end = float(next_item["start"])
        gap = gap_end - gap_start
        if gap <= ADJACENT_BOUNDARY_TOLERANCE:
            continue
        if not _same_song(item, next_item):
            continue

        sentences = _get_item_sentences(item, sentence_cache, cached_asr_results)
        asr_coverage = _sentence_overlap_seconds(gap_start, gap_end, sentences)
        if asr_coverage < max(0.50, min(2.0, gap * 0.20)):
            continue

        template = item if _segment_priority(item) >= _segment_priority(next_item) else next_item
        for part_start, part_end in _split_range_for_gap(gap_start, gap_end, max_duration):
            if part_end - part_start <= BOUNDARY_EPSILON:
                continue
            gap_segment = dict(template)
            gap_segment["start"] = part_start
            gap_segment["end"] = part_end
            gap_segment["is_highlight"] = bool(item.get("is_highlight") or next_item.get("is_highlight"))
            gap_segment["filled_from_asr_gap"] = True
            filled.append(gap_segment)
            fill_count += 1
        if logger:
            logger.warning(
                "[export gap] filled ASR-covered gap %.2fs: %.2f-%.2f asr=%.2fs parts=%s",
                gap,
                gap_start,
                gap_end,
                asr_coverage,
                fill_count,
            )

    return sorted(filled, key=lambda x: (float(x["start"]), float(x["end"]))), fill_count


def _protect_adjacent_sentence_boundaries(
    segments: List[Dict[str, Any]],
    sentence_cache: Dict[int, List[Dict[str, Any]]],
    word_cache: Dict[int, List[Dict[str, Any]]],
    cached_asr_results: Dict[int, dict],
    min_duration: float,
    max_duration: float,
    logger: Optional[logging.Logger],
) -> int:
    fixed_count = 0
    for idx in range(len(segments) - 1):
        prev_item = segments[idx]
        next_item = segments[idx + 1]
        prev_end = float(prev_item["end"])
        next_start = float(next_item["start"])
        gap = next_start - prev_end
        if abs(gap) > ADJACENT_BOUNDARY_TOLERANCE:
            continue

        prev_song = prev_item.get("song")
        next_song = next_item.get("song")
        prev_song_index = int(getattr(prev_song, "song_index", -1))
        next_song_index = int(getattr(next_song, "song_index", -2))
        if prev_song_index != next_song_index:
            continue

        if prev_song_index not in sentence_cache:
            sentence_cache[prev_song_index] = build_global_asr_sentences(prev_song, cached_asr_results)
        if prev_song_index not in word_cache:
            word_cache[prev_song_index] = build_global_asr_words(prev_song, cached_asr_results)

        boundary = (prev_end + next_start) / 2.0
        sentences = sentence_cache.get(prev_song_index, [])
        words = word_cache.get(prev_song_index, [])
        sentence = _find_sentence_containing_boundary(boundary, sentences)
        if sentence:
            cut_point = choose_sentence_complete_adjacent_cut(
                prev_item,
                next_item,
                sentence,
                boundary,
                min_duration,
                max_duration,
                words=words,
                logger=logger,
            )
        else:
            sentence = _find_sentence_ending_near_boundary(boundary, sentences)
            cut_point = (
                choose_sentence_tail_padding_cut(
                    prev_item,
                    next_item,
                    sentence,
                    _next_sentence_start_after(float(sentence["end"]), sentences) if sentence else None,
                    boundary,
                    min_duration,
                    max_duration,
                    logger=logger,
                )
                if sentence
                else None
            )
        if cut_point is None:
            continue

        prev_item["end"] = round(float(cut_point), 3)
        next_item["start"] = round(float(cut_point), 3)
        fixed_count += 1
    return fixed_count


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
    activity_timestamps: Optional[List[Tuple[float, float]]] = None,
    overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Make adjacent export segments non-overlapping and min-duration safe.

    ``activity_timestamps`` is kept for compatibility but intentionally ignored:
    AED activity ranges are classification hints, not safe lyric cut points.
    """
    if len(segments) <= 1:
        return segments

    normalized = [dict(seg) for seg in sorted(segments, key=lambda x: (float(x["start"]), float(x["end"])))]
    _ = activity_timestamps
    cached_asr_results = cached_asr_results or {}
    sentence_cache: Dict[int, List[Dict[str, Any]]] = {}
    word_cache: Dict[int, List[Dict[str, Any]]] = {}
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

    fixed_count += _protect_adjacent_sentence_boundaries(
        normalized,
        sentence_cache,
        word_cache,
        cached_asr_results,
        min_duration,
        max_duration,
        logger,
    )

    final_segments = [
        seg
        for seg in normalized
        if float(seg["end"]) > float(seg["start"]) + BOUNDARY_EPSILON
    ]
    final_segments = _repair_short_segments(
        final_segments,
        min_duration,
        max_duration,
        overlap_tolerance,
        logger,
        sentence_cache,
        cached_asr_results,
    )

    # Duration repair can move a boundary after the subtitle-aware pass. Run one
    # final subtitle validation so the exported media never finalizes a cut that
    # lands inside a cached Doubao sentence.
    fixed_count += _protect_adjacent_sentence_boundaries(
        final_segments,
        sentence_cache,
        word_cache,
        cached_asr_results,
        min_duration,
        max_duration,
        logger,
    )

    final_segments, gap_fill_count = _fill_asr_covered_gaps(
        final_segments,
        sentence_cache,
        cached_asr_results,
        min_duration,
        max_duration,
        logger,
    )
    fixed_count += gap_fill_count

    remaining_overlaps = _remaining_overlaps(final_segments, overlap_tolerance)
    if fixed_count and logger:
        logger.info("[export overlap] normalized %s adjacent overlaps", fixed_count)
    if remaining_overlaps and logger:
        logger.warning("[export overlap] remaining overlaps after normalization: %s", remaining_overlaps[:5])

    return final_segments
