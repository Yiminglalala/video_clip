# -*- coding: utf-8 -*-
"""
手动测试 54-83秒 音频，看看AED为什么没有检测到
"""
import sys, os
sys.path.insert(0, 'D:/video_clip')

import tempfile
import subprocess
import soundfile as sf
from fireredvad import FireRedAed, FireRedAedConfig

print("=" * 80)
print("测试AED检测 54-83秒")
print("=" * 80)

# 初始化AED（用更低的阈值）
aed_config = FireRedAedConfig(
    use_gpu=True,
    smooth_window_size=3,
    speech_threshold=0.05,
    singing_threshold=0.1,
    music_threshold=0.1,
    min_event_frame=5,
    max_event_frame=2000,
    min_silence_frame=10,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000
)

aed = FireRedAed.from_pretrained(
    "D:/video_clip/pretrained_models/xukaituo/FireRedVAD/AED",
    aed_config
)

input_video = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (20).mp4"

# 先提取完整音频
tmp_full = tempfile.mktemp(suffix=".wav")
subprocess.run([
    "ffmpeg", "-y", "-i", input_video,
    "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", tmp_full
], capture_output=True, check=True)

print(f"\n完整音频: {tmp_full}")
wav_full, sr_full = sf.read(tmp_full)
print(f"  时长: {len(wav_full)/sr_full:.1f}秒")

# 测试1: 完整音频
print("\n--- 测试1: 完整音频 ---")
result1, probs1 = aed.detect(tmp_full)
print(f"  singing: {len(result1.get('event2timestamps', {}).get('singing', []))} 段")
print(f"  speech: {len(result1.get('event2timestamps', {}).get('speech', []))} 段")
print(f"  music: {len(result1.get('event2timestamps', {}).get('music', []))} 段")
print(f"  singing_ts: {result1.get('event2timestamps', {}).get('singing', [])[:5]}")
print(f"  speech_ts: {result1.get('event2timestamps', {}).get('speech', [])[:5]}")

# 测试2: 只提取54-83秒
print("\n--- 测试2: 54-83秒 ---")
tmp_54_83 = tempfile.mktemp(suffix=".wav")
subprocess.run([
    "ffmpeg", "-y", "-i", input_video,
    "-ss", "54", "-to", "83",
    "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", tmp_54_83
], capture_output=True, check=True)

print(f"54-83秒音频: {tmp_54_83}")
wav_54, sr_54 = sf.read(tmp_54_83)
print(f"  时长: {len(wav_54)/sr_54:.1f}秒")
print(f"  振幅: min={wav_54.min():.4f}, max={wav_54.max():.4f}, mean={wav_54.mean():.4f}")

result2, probs2 = aed.detect(tmp_54_83)
print(f"\n结果:")
print(f"  singing: {len(result2.get('event2timestamps', {}).get('singing', []))} 段")
print(f"  speech: {len(result2.get('event2timestamps', {}).get('speech', []))} 段")
print(f"  music: {len(result2.get('event2timestamps', {}).get('music', []))} 段")
print(f"  singing_ts: {result2.get('event2timestamps', {}).get('singing', [])}")
print(f"  speech_ts: {result2.get('event2timestamps', {}).get('speech', [])}")

# 清理
os.remove(tmp_full)
os.remove(tmp_54_83)

print("\n" + "=" * 80)
