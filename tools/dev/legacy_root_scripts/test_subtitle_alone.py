#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立测试ASR+字幕烧录"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.processor import LiveVideoProcessor, ProcessingConfig
from pathlib import Path

TEST_VIDEO = r"D:\video_clip\test_video_60s_final.mp4"
OUTPUT_DIR = r"D:\video_clip\output\test_subtitle"

os.makedirs(OUTPUT_DIR, exist_ok=True)

config = ProcessingConfig(
    output_dir=OUTPUT_DIR,
    min_segment_duration=8.0,
    max_segment_duration=15.0,
    enable_songformer=True,
    strict_songformer=True,
    songformer_device="cuda",
    cut_mode="fast",
    enable_subtitle=True,
    subtitle_mode="sentence",
)

processor = LiveVideoProcessor(config)

print("=== [1] 分析视频 ===")
result, export_segments = processor.analyze_video(
    TEST_VIDEO,
    singer=None,
    concert=None,
)

print(f"  ✓ 分析完成: {len(result.songs)}首歌，{len(export_segments)}个片段")
print(f"  ✓ ASR缓存: {len(processor._cached_asr_results)}个key")

if processor._cached_asr_results:
    for k, v in list(processor._cached_asr_results.items())[:3]:
        print(f"    - Key {k}: sentences={len(v.get('sentences', []))}, error={v.get('error')}")

print("\n=== [2] 导出第一个片段 ===")
if export_segments:
    seg = export_segments[0]
    print(f"  ✓ 导出片段: {seg['start']}-{seg['end']}, type={seg['type']}")
    output_files = processor.export_video_segments(
        TEST_VIDEO,
        [seg],
    )
    print(f"  ✓ 输出文件: {output_files}")
else:
    print("  ✗ 没有片段可导出")