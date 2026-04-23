# -*- coding: utf-8 -*-
"""单段测试：只加载一次、只推理一次，避免段错误"""
import numpy as np, librosa, sys, gc
from scipy.signal import butter, sosfilt

# 用完整音频的前14秒
AUDIO_PATH = r'D:\video_clip\output\lyric_aligned\audio.wav'
y, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True, duration=14.0)
print(f"Audio: {len(y)/sr:.1f}s, peak={np.max(np.abs(y)):.4f}", flush=True)

# 预处理
sos = butter(4, 100, btype='high', fs=sr, output='sos')
y_filt = sosfilt(sos, y)
peak = np.max(np.abs(y_filt))
y_proc = (y_filt / peak * 0.9).astype(np.float32)
print(f"Preprocessed: peak={peak:.4f} -> {np.max(np.abs(y_proc)):.4f}", flush=True)

# 加载模型
print("Loading Qwen3-ASR-1.7B ...", flush=True)
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", device_map="cuda:0")
print("Model loaded.", flush=True)

# 释放原始音频
del y, y_filt
gc.collect()

# 只做一次推理
print("\nTranscribing (language=None) ...", flush=True)
result = model.transcribe(audio=(y_proc, 16000), language=None)

print(f"\nResult type: {type(result).__name__}", flush=True)
if isinstance(result, dict):
    print(f"Keys: {list(result.keys())}", flush=True)
    print(f"Text: \"{result.get('text', '')}\"", flush=True)
    print(f"Language: {result.get('language', '?')}", flush=True)
    # 打印原始输出看看有没有隐藏内容
    for k, v in result.items():
        print(f"  {k}: {str(v)[:200]}", flush=True)
else:
    print(f"Str: {str(result)}", flush=True)
