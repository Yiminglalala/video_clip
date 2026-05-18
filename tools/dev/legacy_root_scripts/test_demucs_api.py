# -*- coding: utf-8 -*-
"""
诊断 Demucs 4.0 API 变化
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess, soundfile as sf

# 提取测试音频
tmp_wav = "D:/video_clip/test_demucs_api.wav"
subprocess.run(["ffmpeg", "-y", "-i",
    r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (24).mp4",
    "-t", "20", "-ar", "44100", "-ac", "1", "-acodec", "pcm_s16le", tmp_wav],
    capture_output=True)

# 检查 demucs 版本和 API
import demucs
print("demucs version:", demucs.__version__)

from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch

model = get_model("htdemucs_ft")
print("model sources:", model.sources)
print("model samplerate:", model.samplerate)

# 测试 apply_model
wav, sr = sf.read(tmp_wav, dtype='float32')
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
print("wav_tensor shape:", wav_tensor.shape, "device:", wav_tensor.device)

# 试不同 device
for device in ["auto", "cuda", "cpu"]:
    try:
        print(f"\n--- apply_model device={device} ---")
        result = apply_model(model, wav_tensor, device=device, split=True, overlap=0.5)
        print("result type:", type(result))
        if hasattr(result, '__len__'):
            print("result len:", len(result))
            if isinstance(result, (list, tuple)):
                for i, r in enumerate(result):
                    print(f"  [{i}] shape={getattr(r, 'shape', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

os.remove(tmp_wav)
