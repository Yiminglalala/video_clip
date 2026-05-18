
# -*- coding: utf-8 -*-
"""
测试真实音频的AED检测：原始音频 vs Demucs分离后
"""
import sys
import os
sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.audio_analyzer import VoiceActivityDetector, separate_vocals

# 测试文件
test_video = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (10).mp4"

print("=" * 80)
print("步骤1：从视频提取原始音频")
print("=" * 80)

import tempfile
import subprocess
import soundfile as sf

# 1. 提取原始音频
tmp_original_wav = tempfile.mktemp(suffix='.wav')
cmd = [
    'ffmpeg', '-y',
    '-i', test_video,
    '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
    tmp_original_wav
]
subprocess.run(cmd, check=True, capture_output=True)
print(f"✅ 原始音频提取到: {tmp_original_wav}")

wav, sr = sf.read(tmp_original_wav, dtype='float32')
print(f"✅ 原始音频: sr={sr}, duration={len(wav)/sr:.2f}s")

print("\n" + "=" * 80)
print("步骤2：用原始音频测试AED")
print("=" * 80)

vad = VoiceActivityDetector()
aed_result_original = vad.get_aed_result(tmp_original_wav)

if aed_result_original:
    event2timestamps = aed_result_original.get('event2timestamps', {})
    singing_ts = event2timestamps.get('singing', [])
    speech_ts = event2timestamps.get('speech', [])
    music_ts = event2timestamps.get('music', [])
    print(f"✅ 原始音频AED结果:")
    print(f"  - singing: {len(singing_ts)} 段")
    print(f"  - speech: {len(speech_ts)} 段")
    print(f"  - music: {len(music_ts)} 段")
    if singing_ts:
        print(f"  - singing 前5段: {singing_ts[:5]}")
    if speech_ts:
        print(f"  - speech 前5段: {speech_ts[:5]}")
else:
    print("❌ 原始音频AED返回空")

print("\n" + "=" * 80)
print("步骤3：Demucs人声分离")
print("=" * 80)

demucs_dir = tempfile.mkdtemp(prefix="dmucs_test_")
vocals_path, no_vocals_path = separate_vocals(
    test_video,
    out_dir=demucs_dir,
    model="htdemucs_ft"
)

if vocals_path and os.path.exists(vocals_path):
    print(f"✅ Demucs分离成功:")
    print(f"  - vocals: {vocals_path}")
    wav_vocals, sr_vocals = sf.read(vocals_path, dtype='float32')
    print(f"  - vocals: duration={len(wav_vocals)/sr_vocals:.2f}s, max={wav_vocals.max():.4f}, mean={abs(wav_vocals).mean():.4f}")
else:
    print("❌ Demucs分离失败")

print("\n" + "=" * 80)
print("步骤4：用Demucs分离后的音频测试AED")
print("=" * 80)

if vocals_path and os.path.exists(vocals_path):
    aed_result_vocals = vad.get_aed_result(vocals_path)
    
    if aed_result_vocals:
        event2timestamps_v = aed_result_vocals.get('event2timestamps', {})
        singing_ts_v = event2timestamps_v.get('singing', [])
        speech_ts_v = event2timestamps_v.get('speech', [])
        music_ts_v = event2timestamps_v.get('music', [])
        print(f"✅ Demucs分离后AED结果:")
        print(f"  - singing: {len(singing_ts_v)} 段")
        print(f"  - speech: {len(speech_ts_v)} 段")
        print(f"  - music: {len(music_ts_v)} 段")
        if singing_ts_v:
            print(f"  - singing 前5段: {singing_ts_v[:5]}")
        if speech_ts_v:
            print(f"  - speech 前5段: {speech_ts_v[:5]}")
    else:
        print("❌ Demucs分离后AED返回空")

print("\n" + "=" * 80)
print("清理临时文件")
print("=" * 80)

import shutil
os.remove(tmp_original_wav)
if os.path.exists(demucs_dir):
    shutil.rmtree(demucs_dir)
    print(f"✅ 已清理: {demucs_dir}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

