# -*- coding: utf-8 -*-
"""
检查歌词字幕链路运行环境。

用法：
    python scripts/check_lyric_subtitle_env.py

建议使用 UI 同一个解释器：
    SongFormer_install\\venv_gpu\\Scripts\\python.exe scripts\\check_lyric_subtitle_env.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REQUIRED = [
    "streamlit",
    "torch",
    "torchaudio",
    "librosa",
    "soundfile",
    "requests",
    "shazamio",
    "ShazamAPI",
    "whisper",
]

OPTIONAL = [
    "whisperx",
    "faster_whisper",
]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_ffmpeg_probe(binary: str) -> str:
    try:
        out = subprocess.check_output([binary, "-version"], text=True, stderr=subprocess.STDOUT)
        return out.splitlines()[0]
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> int:
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print()

    ok = True
    print("[Required Python packages]")
    for name in REQUIRED:
        present = has_module(name)
        ok = ok and present
        print(f"  {name}: {'OK' if present else 'MISSING'}")

    print()
    print("[Optional Python packages]")
    for name in OPTIONAL:
        present = has_module(name)
        print(f"  {name}: {'OK' if present else 'MISSING'}")

    print()
    print("[Tools]")
    print(f"  ffmpeg:  {run_ffmpeg_probe('ffmpeg')}")
    print(f"  ffprobe: {run_ffmpeg_probe('ffprobe')}")

    print()
    print("[Torch]")
    try:
        import torch

        print(f"  torch: {torch.__version__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        print(f"  cuda version: {getattr(torch.version, 'cuda', None)}")
    except Exception as exc:
        ok = False
        print(f"  ERROR: {exc}")

    print()
    print("[Model cache]")
    hf_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_roots = [
        hf_root / "models--Systran--faster-whisper-tiny",
        hf_root / "models--Systran--faster-whisper-small",
        hf_root / "models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn",
    ]
    for root in model_roots:
        files = list(root.glob("snapshots/*/*")) if root.exists() else []
        print(f"  {root.name}: {'OK' if files else 'MISSING'} ({len(files)} files)")

    print()
    summary = {"ready": ok, "optional_whisperx": has_module("whisperx")}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
