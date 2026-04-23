# -*- coding: utf-8 -*-
"""
官方AED测试 - 对比原始音频 vs Demucs分离后的音频
"""
import sys, os
sys.path.insert(0, 'D:/video_clip')

import soundfile as sf
from fireredvad import FireRedAed, FireRedAedConfig

print("=" * 80)
print("官方AED测试 - 对比原始音频 vs Demucs分离后的音频")
print("=" * 80)

# 初始化AED（用官方默认参数）
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

# 测试1: 原始音频（从视频提取）
print("\n" + "=" * 80)
print("测试1: 原始音频（不用Demucs）")
print("=" * 80)

original_video = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (6).mp4"

import tempfile
import subprocess

tmp_original = tempfile.mktemp(suffix=".wav")
subprocess.run([
    "ffmpeg", "-y", "-i", original_video,
    "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", tmp_original
], capture_output=True, check=True)

print(f"原始音频: {tmp_original}")
wav_orig, sr_orig = sf.read(tmp_original)
print(f"  时长: {len(wav_orig)/sr_orig:.1f}秒")
print(f"  采样率: {sr_orig}Hz")
print(f"  通道数: {wav_orig.ndim}")

result1, probs1 = aed.detect(tmp_original)
print(f"\n结果1（原始音频）:")
print(f"  singing: {len(result1.get('event2timestamps', {}).get('singing', []))} 段")
print(f"  speech: {len(result1.get('event2timestamps', {}).get('speech', []))} 段")
print(f"  music: {len(result1.get('event2timestamps', {}).get('music', []))} 段")
if result1.get('event2timestamps', {}).get('singing', []):
    print(f"  singing时间戳 (前3个): {result1['event2timestamps']['singing'][:3]}")
if result1.get('event2timestamps', {}).get('speech', []):
    print(f"  speech时间戳 (前3个): {result1['event2timestamps']['speech'][:3]}")
if result1.get('event2timestamps', {}).get('music', []):
    print(f"  music时间戳 (前3个): {result1['event2timestamps']['music'][:3]}")

# 测试2: Demucs分离后的vocals.wav
print("\n" + "=" * 80)
print("测试2: Demucs分离后的vocals.wav")
print("=" * 80)

demucs_vocals = r"D:\video_clip\output\demucs_test\vocals.wav"
if os.path.exists(demucs_vocals):
    print(f"Demucs vocals: {demucs_vocals}")
    wav_dm, sr_dm = sf.read(demucs_vocals)
    print(f"  时长: {len(wav_dm)/sr_dm:.1f}秒")
    print(f"  采样率: {sr_dm}Hz")
    print(f"  通道数: {wav_dm.ndim}")
    
    result2, probs2 = aed.detect(demucs_vocals)
    print(f"\n结果2（Demucs分离后）:")
    print(f"  singing: {len(result2.get('event2timestamps', {}).get('singing', []))} 段")
    print(f"  speech: {len(result2.get('event2timestamps', {}).get('speech', []))} 段")
    print(f"  music: {len(result2.get('event2timestamps', {}).get('music', []))} 段")
    if result2.get('event2timestamps', {}).get('singing', []):
        print(f"  singing时间戳 (前3个): {result2['event2timestamps']['singing'][:3]}")
    if result2.get('event2timestamps', {}).get('speech', []):
        print(f"  speech时间戳 (前3个): {result2['event2timestamps']['speech'][:3]}")
    if result2.get('event2timestamps', {}).get('music', []):
        print(f"  music时间戳 (前3个): {result2['event2timestamps']['music'][:3]}")
else:
    print(f"文件不存在: {demucs_vocals}")

# 清理
os.remove(tmp_original)

print("\n" + "=" * 80)
print("测试完成!")
print("=" * 80)
