"""Central project path definitions.

Keep user-facing outputs, runtime cache, QA artifacts, and local backups in
separate folders so new files do not accumulate in the repository root.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "output"
VIDEO_OUTPUT_DIR = OUTPUT_ROOT / "videos"
SUBTITLE_OUTPUT_DIR = OUTPUT_ROOT / "subtitles"
OUTPUT_CACHE_DIR = OUTPUT_ROOT / "cache"
QA_OUTPUT_DIR = OUTPUT_ROOT / "qa"
PLAYWRIGHT_OUTPUT_DIR = QA_OUTPUT_DIR / "playwright"
SUBTITLE_PROBE_DIR = QA_OUTPUT_DIR / "subtitle_probe"

TEMP_ROOT = PROJECT_ROOT / "temp"
LOG_DIR = PROJECT_ROOT / "logs"
LOCAL_STATE_DIR = PROJECT_ROOT / ".local_state"
BACKUP_DIR = PROJECT_ROOT / "backups"


def ensure_project_dirs() -> None:
    """Create the standard writable project directories."""
    for path in (
        OUTPUT_ROOT,
        VIDEO_OUTPUT_DIR,
        SUBTITLE_OUTPUT_DIR,
        OUTPUT_CACHE_DIR,
        QA_OUTPUT_DIR,
        PLAYWRIGHT_OUTPUT_DIR,
        SUBTITLE_PROBE_DIR,
        TEMP_ROOT,
        LOG_DIR,
        LOCAL_STATE_DIR,
        BACKUP_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
