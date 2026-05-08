# -*- coding: utf-8 -*-
"""Small Streamlit-facing helpers for the slicing workflow."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from src.output_spec import DEFAULT_LANDSCAPE_RESOLUTION, normalize_landscape_resolution_choice
from src.processor import ProcessingConfig


def detect_songformer_device() -> Tuple[str, str]:
    """Return the preferred SongFormer device plus a user-facing status message."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_mem_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            gpu_mem = gpu_mem_bytes / (1024**3)
            return "cuda", f"GPU 可用：{gpu_name} ({gpu_mem:.1f}GB)，模型将使用 CUDA 加速"
        return "cpu", "CUDA 不可用：SongFormer / Demucs / FireRed 将以 CPU 运行，速度会显著下降。"
    except Exception as exc:
        return "auto", f"无法检测 CUDA 状态：{exc}，将尝试 auto 模式。"


def build_slicing_processing_config(
    config_dict: Dict[str, Any],
    output_dir: str,
    songformer_device: str,
    source_orientation: str = "auto",
    doubao_appid: str = "",
    doubao_access_token: str = "",
) -> ProcessingConfig:
    return ProcessingConfig(
        output_dir=output_dir,
        min_segment_duration=float(config_dict.get("min_dur", 8.0)),
        max_segment_duration=float(config_dict.get("max_dur", 15.0)),
        min_duration_limit=float(config_dict.get("min_dur", 8.0)),
        max_duration_limit=float(config_dict.get("max_dur", 15.0)),
        enable_songformer=True,
        strict_songformer=True,
        songformer_device=songformer_device,
        songformer_window=60,
        songformer_hop=30,
        concert=config_dict.get("concert_name") or None,
        cut_mode=config_dict.get("cut_mode", "fast"),
        enable_subtitle=bool(config_dict.get("enable_subtitle", False)),
        subtitle_mode="sentence",
        landscape_resolution_choice=normalize_landscape_resolution_choice(
            config_dict.get("landscape_resolution_choice", DEFAULT_LANDSCAPE_RESOLUTION)
        ),
        source_orientation=source_orientation,
        doubao_appid=doubao_appid,
        doubao_access_token=doubao_access_token,
    )
