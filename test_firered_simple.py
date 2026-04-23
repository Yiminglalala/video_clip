
# -*- coding: utf-8 -*-
"""
简单测试：看看 FireRedVAD/AED 实际返回什么
"""
import sys
import os
sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 1. 先直接加载 VAD
from fireredvad import FireRedVad, FireRedVadConfig
_FIREDRED_VAD_MODEL_DIR = "D:/video_clip/pretrained_models/xukaituo/FireRedVAD/VAD"

print("=" * 80)
print("测试 FireRedVad")
print("=" * 80)

# 加载模型
vad_model = FireRedVad.from_pretrained(
    _FIREDRED_VAD_MODEL_DIR,
    FireRedVadConfig(use_gpu=True)
)

# 创建一个简单的测试音频
import numpy as np
sr = 16000
wav = np.random.randn(sr * 5).astype(np.float32)

# 测试 detect
result = vad_model.detect(wav, sr)
print(f"VAD 返回类型: {type(result)}")
if isinstance(result, tuple):
    print(f"  是 tuple，长度: {len(result)}")
    for i, item in enumerate(result):
        print(f"  [{i}] type={type(item)}, value={item}")
        if hasattr(item, 'keys'):
            print(f"  [{i}] keys={list(item.keys())}")
else:
    print(f"VAD 结果: {result}")
    if hasattr(result, 'keys'):
        print(f"  keys={list(result.keys())}")

print("\n" + "=" * 80)
print("测试 FireRedAed")
print("=" * 80)

from fireredvad import FireRedAed, FireRedAedConfig
_FIREDRED_AED_MODEL_DIR = "D:/video_clip/pretrained_models/xukaituo/FireRedVAD/AED"

aed_model = FireRedAed.from_pretrained(
    _FIREDRED_AED_MODEL_DIR,
    FireRedAedConfig(use_gpu=True)
)

result, info = aed_model.detect((wav, sr))
print(f"AED 返回类型: {type(result)}")
if isinstance(result, tuple):
    print(f"  是 tuple，长度: {len(result)}")
    for i, item in enumerate(result):
        print(f"  [{i}] type={type(item)}, value={item}")
        if hasattr(item, 'keys'):
            print(f"  [{i}] keys={list(item.keys())}")
else:
    print(f"AED 结果: {result}")
    if hasattr(result, 'keys'):
        print(f"  keys={list(result.keys())}")

print("\n" + "=" * 80)

