"""Path helpers shared by Streamlit UI entrypoints."""

from __future__ import annotations


_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "`": "`",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "《": "》",
}

_QUOTE_CHARS = set(_QUOTE_PAIRS) | set(_QUOTE_PAIRS.values())


def normalize_pasted_local_path(raw_path: object) -> str:
    """Normalize a local path pasted from Explorer, chat apps, or terminals."""
    if raw_path is None:
        return ""

    path = str(raw_path).strip().lstrip("\ufeff")
    if not path:
        return ""

    # Windows "Copy as path" commonly wraps paths in quotes.
    for _ in range(4):
        path = path.strip()
        if len(path) < 2:
            break

        first, last = path[0], path[-1]
        if _QUOTE_PAIRS.get(first) == last:
            path = path[1:-1]
            continue

        if first in _QUOTE_CHARS and last in _QUOTE_CHARS:
            path = path[1:-1]
            continue

        break

    return path.strip()
