# -*- coding: utf-8 -*-
"""Shared subtitle export formatting helpers."""

from __future__ import annotations


DEFAULT_SUBTITLE_MAX_CHARS = 10
DEFAULT_SUBTITLE_TARGET_CHARS = 8
DEFAULT_SUBTITLE_MIN_TAIL_CHARS = 4
SUBTITLE_SPLIT_PUNCTUATIONS = "。！？、，；：,.;:!? "


def find_subtitle_split_index(text: str, max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS) -> int:
    value = str(text or "")
    if len(value) <= max_chars:
        return len(value)

    search_start = max(1, max_chars - 4)
    search_end = min(len(value) - 1, max_chars)
    for idx in range(search_end, search_start - 1, -1):
        if value[idx - 1] in SUBTITLE_SPLIT_PUNCTUATIONS:
            return idx

    return min(max_chars, len(value) - 1)


def _balanced_subtitle_chunks(
    text: str,
    max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS,
    target_chars: int = DEFAULT_SUBTITLE_TARGET_CHARS,
    min_tail_chars: int = DEFAULT_SUBTITLE_MIN_TAIL_CHARS,
) -> list[str]:
    value = str(text or "").strip()
    if not value or len(value) <= max_chars:
        return [value] if value else []

    total_len = len(value)
    chunk_count = max(2, (total_len + target_chars - 1) // target_chars)
    while (total_len + chunk_count - 1) // chunk_count > max_chars:
        chunk_count += 1

    base = total_len // chunk_count
    extra = total_len % chunk_count
    sizes = [base + (1 if idx < extra else 0) for idx in range(chunk_count)]

    while len(sizes) > 1 and sizes[-1] < min_tail_chars and sizes[-2] < max_chars:
        sizes[-2] += 1
        sizes[-1] -= 1
        if sizes[-1] <= 0:
            sizes.pop()
            break

    chunks: list[str] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        if cursor >= total_len:
            break
        remaining_chunks = len(sizes) - idx - 1
        ideal_end = total_len if remaining_chunks == 0 else cursor + size
        min_end = cursor + max(1, min(size, max_chars) - 2)
        max_end = min(cursor + max_chars, total_len - remaining_chunks)

        split_end = ideal_end
        for candidate in range(min(max_end, ideal_end + 2), max(cursor + 1, min_end) - 1, -1):
            if value[candidate - 1] in SUBTITLE_SPLIT_PUNCTUATIONS:
                split_end = candidate
                break

        split_end = max(cursor + 1, min(max_end, split_end))
        chunk = value[cursor:split_end].strip()
        if chunk:
            chunks.append(chunk)
        cursor = split_end

    if cursor < total_len:
        tail = value[cursor:].strip()
        if tail:
            chunks.append(tail)

    if len(chunks) > 1 and len(chunks[-1]) < min_tail_chars and len(chunks[-2]) + len(chunks[-1]) <= max_chars:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()

    if any(len(chunk) > max_chars for chunk in chunks):
        fallback = []
        cursor = 0
        while cursor < total_len:
            fallback.append(value[cursor:cursor + max_chars])
            cursor += max_chars
        chunks = fallback

    return chunks


def split_subtitle_sentence_entry(sent: dict, max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS) -> list[dict]:
    text = str(sent.get("text", "") or "").strip()
    if not text or len(text) <= max_chars:
        return [sent]

    chunks = _balanced_subtitle_chunks(text, max_chars=max_chars)

    if len(chunks) <= 1:
        return [sent]

    start = float(sent.get("start", 0.0) or 0.0)
    end = float(sent.get("end", start + 3.0) or (start + 3.0))
    total_duration = max(end - start, 0.0)
    total_chars = max(sum(len(chunk) for chunk in chunks), 1)
    cursor = start
    result: list[dict] = []

    for idx, chunk in enumerate(chunks):
        item = dict(sent)
        item["text"] = chunk
        item["start"] = cursor
        if idx == len(chunks) - 1:
            item["end"] = end
        elif total_duration <= 0.12:
            item["end"] = min(end, cursor + 0.06)
        else:
            item_duration = total_duration * (len(chunk) / total_chars)
            item["end"] = min(end, max(cursor + 0.06, cursor + item_duration))
        cursor = float(item["end"])
        result.append(item)

    return result


def format_ass_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def escape_ass_text(text: object) -> str:
    value = str(text or "")
    value = value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return value.replace("\n", "\\N").replace("\r", "")
