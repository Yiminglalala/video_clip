# -*- coding: utf-8 -*-
"""Runtime-only configuration helpers.

This module keeps private credentials out of source code. Values are read from
environment variables first, then from ``local_config.json`` at the project root.
The local config file is intentionally git-ignored.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_CONFIG_PATH = PROJECT_ROOT / "local_config.json"


def get_runtime_config_path() -> Path:
    """Return the local private runtime config path."""
    raw = os.environ.get("VIDEO_CLIP_LOCAL_CONFIG", "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_LOCAL_CONFIG_PATH


@lru_cache(maxsize=1)
def load_local_runtime_config() -> Dict[str, Any]:
    """Load private local runtime config if present."""
    path = get_runtime_config_path()
    if not path.exists():
        return {}
    try:
        # Windows PowerShell may create UTF-8 files with BOM; utf-8-sig accepts both.
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _lookup_config_key(data: Dict[str, Any], key: str) -> str:
    """Lookup either flat keys or dotted nested keys from local config."""
    if not data or not key:
        return ""

    if key in data and data[key] not in (None, ""):
        return str(data[key]).strip()

    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return ""
        node = node[part]
    return str(node).strip() if node not in (None, "") else ""


def get_runtime_value(env_name: str, *config_keys: str, default: str = "") -> str:
    """Resolve a runtime value from env first, then local config."""
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return str(env_value).strip()

    data = load_local_runtime_config()
    for key in config_keys:
        value = _lookup_config_key(data, key)
        if value:
            return value
    return default


def get_doubao_credentials() -> Tuple[str, str]:
    """Return Doubao API credentials from private runtime config."""
    appid = get_runtime_value("DOUBAO_APPID", "doubao.appid", "doubao_appid")
    access_token = get_runtime_value(
        "DOUBAO_ACCESS_TOKEN",
        "doubao.access_token",
        "doubao_access_token",
    )
    return appid, access_token


def has_doubao_credentials() -> bool:
    appid, access_token = get_doubao_credentials()
    return bool(appid and access_token)
