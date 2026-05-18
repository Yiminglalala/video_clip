#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查缓存机制是否工作"""
import sys
import os
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.processor import LiveVideoProcessor, ProcessingConfig

TEST_VIDEO = r"D:\video_clip\test_video_60s_final.mp4"

print("=== [1] 创建临时config ===")
config = ProcessingConfig(
    output_dir=r"D:\video_clip\output\test_cache",
    min_segment_duration=8.0,
    max_segment_duration=15.0,
    enable_songformer=True,
    strict_songformer=True,
    songformer_device="cuda",
    cut_mode="fast",
    enable_subtitle=True,
    subtitle_mode="sentence",
)

print("=== [2] 创建processor ===")
processor = LiveVideoProcessor(config)
print(f"  ✓ 初始缓存: {len(processor._cached_asr_results)}")

print("\n=== [3] 模拟设置缓存 ===")
test_asr = {
    0: {
        "sentences": [{"text": "这是测试字幕", "start": 0, "end": 5}],
        "words": [],
        "error": None,
    }
}
processor.set_cached_asr_results(test_asr)
print(f"  ✓ 设置后缓存: {len(processor._cached_asr_results)}")
for k, v in processor._cached_asr_results.items():
    print(f"    - key {k}: {len(v['sentences'])} sentences")

print("\n=== [4] 检查是否能读取 ===")
print(f"  ✓ 缓存类型: {type(processor._cached_asr_results)}")
print(f"  ✓ 缓存内容: {processor._cached_asr_results}")

print("\n=== [5] 检查processor.py的set_cached_asr_results方法 ===")
with open(r"D:\video_clip\src\processor.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
found = False
for i, line in enumerate(lines[100:130], 100):
    if "def set_cached_asr_results" in line:
        found = True
        print(f"  ✓ 在第 {i+1} 行找到 set_cached_asr_results")
        for j in range(i, i+10):
            if j < len(lines):
                print(f"    {j+1}: {lines[j].rstrip()}")
        break
if not found:
    print("  ✗ 未找到 set_cached_asr_results")