# -*- coding: utf-8 -*-
"""
最简单的官方AED测试
"""
import sys, os
sys.path.insert(0, 'D:/video_clip')

import soundfile as sf
from fireredvad import FireRedAed, FireRedAedConfig

print("=" * 80)
print("官方AED测试 - 不用Demucs")
print("=" * 80)

# 初始化AED
aed_config = FireRedAedConfig(
    use_gpu=True,
    smooth_window_size=5,
    speech_threshold=0.4,
    singing_threshold=0.5,
    music_threshold=0.5,
    min_event_frame=20,
    max_event_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000
)

aed = FireRedAed.from_pretrained(
    "D:/video_clip/pretrained_models/xukaituo/FireRedVAD/AED",
    aed_config
)

# 测试文件 - 从之前的输出找一个vocals.wav
test_vocals = r"D:\video_clip\output\demucs_test\vocals.wav"

if os.path.exists(test_vocals):
    print(f"\n测试文件: {test_vocals}")
    result, probs = aed.detect(test_vocals)
    print(f"\n结果:")
    print(f"  singing: {len(result.get('event2timestamps', {}).get('singing', []))} 段")
    print(f"  speech: {len(result.get('event2timestamps', {}).get('speech', []))} 段")
    print(f"  music: {len(result.get('event2timestamps', {}).get('music', []))} 段")
    
    print(f"\nsinging时间戳: {result.get('event2timestamps', {}).get('singing', [])[:5]}")
    print(f"speech时间戳: {result.get('event2timestamps', {}).get('speech', [])[:5]}")
    print(f"music时间戳: {result.get('event2timestamps', {}).get('music', [])[:5]}")
else:
    print(f"文件不存在: {test_vocals}")

print("\n" + "=" * 80)
