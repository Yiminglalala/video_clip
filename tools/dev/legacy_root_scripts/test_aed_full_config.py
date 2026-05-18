# -*- coding: utf-8 -*-
"""
用完整流程的AED配置测试完整音频
"""
import sys, os
sys.path.insert(0, 'D:/video_clip')

import tempfile
import subprocess
import soundfile as sf
from fireredvad import FireRedAed, FireRedAedConfig

print("=" * 80)
print("用完整流程的AED配置测试完整音频")
print("=" * 80)

# 用audio_analyzer.py中的完整配置
aed_config = FireRedAedConfig(
    use_gpu=True,
    speech_threshold=0.1,
    singing_threshold=0.2,
    music_threshold=0.2,
    smooth_window_size=3,
    min_event_frame=10,
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

# 测试完整音频
print("\n--- 完整音频检测 ---")
result, probs = aed.detect(tmp_full)
print(f"  singing: {len(result.get('event2timestamps', {}).get('singing', []))} 段")
print(f"  speech: {len(result.get('event2timestamps', {}).get('speech', []))} 段")
print(f"  music: {len(result.get('event2timestamps', {}).get('music', []))} 段")

singing_ts = result.get('event2timestamps', {}).get('singing', [])
speech_ts = result.get('event2timestamps', {}).get('speech', [])

print(f"\nsinging_ts:")
for ts in singing_ts:
    print(f"  {ts[0]:.2f} - {ts[1]:.2f}")

print(f"\nspeech_ts:")
for ts in speech_ts:
    print(f"  {ts[0]:.2f} - {ts[1]:.2f}")

# 清理
os.remove(tmp_full)

print("\n" + "=" * 80)
