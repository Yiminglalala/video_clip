# -*- coding: utf-8 -*-
"""
用官方方式测试 AED，对比：
1. 原始音频
2. Demucs分离后的vocals.wav
"""
import sys, os
sys.path.insert(0, 'D:/video_clip')

import soundfile as sf
from fireredvad import FireRedAed, FireRedAedConfig

print("=" * 80)
print("测试1: 用原始音频")
print("=" * 80)

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

# 测试1: 原始音频
print("\n--- 测试1: 原始音频 ---")
original_audio = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (6).mp4"
# 先用ffmpeg提取音频
import tempfile
import subprocess
tmp_wav = tempfile.mktemp(suffix=".wav")
subprocess.run([
    "ffmpeg", "-y", "-i", original_audio,
    "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", tmp_wav
], capture_output=True, check=True)

print(f"  提取原始音频: {tmp_wav}")
result1, probs1 = aed.detect(tmp_wav)
print(f"  singing: {len(result1.get('event2timestamps', {}).get('singing', []))} 段")
print(f"  speech: {len(result1.get('event2timestamps', {}).get('speech', []))} 段")
print(f"  music: {len(result1.get('event2timestamps', {}).get('music', []))} 段")

# 测试2: Demucs分离后的音频
print("\n--- 测试2: Demucs分离后的vocals.wav ---")
demucs_vocals = r"D:\video_clip\output\demucs_test\vocals.wav"
if os.path.exists(demucs_vocals):
    print(f"  使用Demucs vocals: {demucs_vocals}")
    result2, probs2 = aed.detect(demucs_vocals)
    print(f"  singing: {len(result2.get('event2timestamps', {}).get('singing', []))} 段")
    print(f"  speech: {len(result2.get('event2timestamps', {}).get('speech', []))} 段")
    print(f"  music: {len(result2.get('event2timestamps', {}).get('music', []))} 段")
else:
    print(f"  Demucs vocals不存在: {demucs_vocals}")

# 清理
os.remove(tmp_wav)

print("\n" + "=" * 80)
print("测试完成!")
print("=" * 80)
