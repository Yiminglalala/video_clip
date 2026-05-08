# -*- coding: utf-8 -*-
"""Shared subtitle export formatting helpers."""

from __future__ import annotations


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
