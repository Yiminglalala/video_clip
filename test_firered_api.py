
# -*- coding: utf-8 -*-
"""
验证 FireRedVAD/AED 的正确 API 用法
"""
import sys
import os
sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 先直接测试 FireRedVAD 包
print("=" * 80)
print("直接测试 fireredvad 包")
print("=" * 80)

try:
    from fireredvad import FireRedVad, FireRedVadConfig, FireRedAed, FireRedAedConfig
    
    # 测试文件
    test_audio = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (10).mp4"
    
    # 1. 提取原始音频
    import tempfile
    import subprocess
    import soundfile as sf
    
    tmp_wav = tempfile.mktemp(suffix='.wav')
    cmd = [
        'ffmpeg', '-y',
        '-i', test_audio,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        tmp_wav
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✅ 提取音频到: {tmp_wav}")
    
    wav, sr = sf.read(tmp_wav, dtype='float32')
    print(f"✅ 加载音频: sr={sr}, duration={len(wav)/sr:.2f}s")
    
    # 2. 直接用 FireRedVad
    print("\n🔍 测试 FireRedVad:")
    from fireredvad import FireRedVad, FireRedVadConfig
    _FIREDRED_VAD_MODEL_DIR = "D:/video_clip/pretrained_models/FireRedVAD/VAD"
    vad_model = FireRedVad.from_pretrained(
        _FIREDRED_VAD_MODEL_DIR,
        FireRedVadConfig(use_gpu=True)
    )
    vad_result = vad_model.detect(wav, sr)
    print(f"✅ VAD 返回类型: {type(vad_result)}")
    if isinstance(vad_result, tuple):
        print(f"  VAD 是 tuple, 长度: {len(vad_result)}")
        for i, item in enumerate(vad_result):
            print(f"  [{i}] type={type(item)}")
            if hasattr(item, 'keys'):
                print(f"  [{i}] keys={list(item.keys())}")
    else:
        print(f"  VAD 结果: {vad_result}")
    
    # 3. 直接用 FireRedAed
    print("\n🔍 测试 FireRedAed:")
    _FIREDRED_AED_MODEL_DIR = "D:/video_clip/pretrained_models/FireRedVAD/AED"
    aed_model = FireRedAed.from_pretrained(
        _FIREDRED_AED_MODEL_DIR,
        FireRedAedConfig(use_gpu=True)
    )
    aed_result, aed_info = aed_model.detect((wav, sr))
    print(f"✅ AED 返回类型: {type(aed_result)}")
    if isinstance(aed_result, tuple):
        print(f"  AED 是 tuple, 长度: {len(aed_result)}")
        for i, item in enumerate(aed_result):
            print(f"  [{i}] type={type(item)}")
            if hasattr(item, 'keys'):
                print(f"  [{i}] keys={list(item.keys())}")
    else:
        print(f"  AED 结果: {aed_result}")
    
    event2timestamps = aed_result.get('event2timestamps', {})
    singing_ts = event2timestamps.get('singing', [])
    speech_ts = event2timestamps.get('speech', [])
    music_ts = event2timestamps.get('music', [])
    print(f"✅ AED 检测结果:")
    print(f"  - singing: {len(singing_ts)} 段")
    print(f"  - speech: {len(speech_ts)} 段")
    print(f"  - music: {len(music_ts)} 段")
    
    # 清理
    os.remove(tmp_wav)
    
except Exception as e:
    import traceback
    print(f"❌ 错误: {e}")
    print(traceback.format_exc())

print("\n" + "=" * 80)

