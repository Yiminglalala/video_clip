# -*- coding: utf-8 -*-
"""
直接测试 Demucs 4.0 apply_model 真实返回值
"""
import sys, os
sys.path.insert(0, 'D:/video_clip/src')

import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

tmp = "D:/video_clip/test_demucs_verify.wav"
import subprocess
subprocess.run(["ffmpeg", "-y", "-i",
    r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (24).mp4",
    "-t", "10", "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le", tmp],
    capture_output=True)

wav, sr = sf.read(tmp, dtype='float32')
wav = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)  # [2, T]
wav = wav.unsqueeze(0)  # [1, 2, T]
print("input shape:", wav.shape)

model = get_model("htdemucs_ft")
print("sources:", model.sources, "samplerate:", model.samplerate)

# test apply_model
with torch.no_grad():
    result = apply_model(model, wav, device="cuda", shifts=1, split=True, overlap=0.1, progress=False, segment=7)

print("result type:", type(result))
print("result len:", len(result))
for i, r in enumerate(result):
    print(f"  [{i}] type={type(r)}, ", end="")
    if hasattr(r, 'shape'):
        print(f"shape={r.shape}, dtype={r.dtype}")
    elif isinstance(r, (list, tuple)):
        print(f"len={len(r)}, inner types={[type(x) for x in r]}")
    else:
        print(f"value={str(r)[:100]}")

# 尝试不同的调用方式
print("\n--- 测试不分块 ---")
with torch.no_grad():
    result2 = apply_model(model, wav, device="cuda", shifts=0, split=False, progress=False)
print("result2 type:", type(result2), "len:", len(result2))

print("\n--- 测试 CPU ---")
with torch.no_grad():
    result3 = apply_model(model, wav, device="cpu", shifts=0, split=False, progress=False)
print("result3 type:", type(result3), "len:", len(result3))
if hasattr(result3, 'shape'):
    print("result3 shape:", result3.shape)

os.remove(tmp)
