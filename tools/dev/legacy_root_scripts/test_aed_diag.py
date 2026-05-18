import os
import sys
import soundfile as sf
import librosa
import tempfile

sys.path.insert(0, 'D:/video_clip')

from src.audio_analyzer import separate_vocals

test_audio = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (24).mp4"

print("Step 1: 提取音频...")
from src.ffmpeg_processor import extract_audio
wav_path = extract_audio(test_audio)
print(f"  音频路径: {wav_path}")

print("\nStep 2: Demucs分离...")
vocals_path, no_vocals_path = separate_vocals(wav_path, out_dir="D:/video_clip/demucs_check")
print(f"  vocals: {vocals_path}")
print(f"  no_vocals: {no_vocals_path}")

print("\nStep 3: 检查vocals.wav...")
wav, sr = sf.read(vocals_path)
print(f"  shape: {wav.shape}")
print(f"  dtype: {wav.dtype}")
print(f"  sr: {sr}")
print(f"  ndim: {wav.ndim}")
if wav.ndim > 1:
    print(f"  通道数: {wav.shape[1]}")

print("\nStep 4: 测试 AED get_aed_result...")
from src.audio_analyzer import AudioAnalyzer
aa = AudioAnalyzer()
result = aa.vad.get_aed_result(vocals_path)
print(f"  result type: {type(result)}")
if result:
    print(f"  keys: {list(result.keys())}")
    print(f"  event2timestamps: {list(result.get('event2timestamps', {}).keys())}")
else:
    print("  result is None!")

print("\nStep 5: 直接调用 _firered_aed_model.detect...")
import soundfile as sf
wav2, sr2 = sf.read(vocals_path, dtype='float32')
if wav2.ndim > 1:
    wav2 = wav2.mean(axis=1)

print(f"  处理后 shape: {wav2.shape}, sr={sr2}")

from fireredvad.core import get_aed_model
_firered_aed_model = get_aed_model()

import torch
print(f"  模型已加载: {_firered_aed_model}")

try:
    result2, _ = _firered_aed_model.detect((wav2, sr2))
    print(f"  直接检测结果: {type(result2)}")
    if result2:
        print(f"  event2timestamps keys: {list(result2.get('event2timestamps', {}).keys())}")
        for k, v in result2.get('event2timestamps', {}).items():
            print(f"    {k}: {len(v)}段")
except Exception as e:
    import traceback
    print(f"  Error: {e}")
    print(f"  Traceback: {traceback.format_exc()}")