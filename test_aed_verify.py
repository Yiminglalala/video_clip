
# -*- coding: utf-8 -*-
"""
测试 AED 在原始音频 vs Demucs分离后音频上的表现
"""
import sys
import os
sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.audio_analyzer import VoiceActivityDetector

# 测试文件
original_audio = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (10).mp4"
# 从之前的测试中提取的Demucs分离后的音频路径示例（需要实际存在的路径）
# demucs_vocals = r"C:\Users\YIMING\AppData\Local\Temp\dmucs_xxxx\vocals.wav"

vad = VoiceActivityDetector()

print("=" * 80)
print("测试 AED 在原始音频上的表现")
print("=" * 80)

try:
    import soundfile as sf
    import tempfile
    
    # 1. 先从视频提取原始音频
    import tempfile
    import subprocess
    import tempfile
    
    tmp_wav = tempfile.mktemp(suffix='.wav')
    cmd = [
        'ffmpeg', '-y',
        '-i', original_audio,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1',
        tmp_wav
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✅ 提取原始音频到: {tmp_wav}")
    
    # 2. 用原始音频测试AED
    print("\n🔍 测试原始音频 AED:")
    aed_result = vad.get_aed_result(tmp_wav)
    if aed_result:
        singing_ts = aed_result.get('event2timestamps', {}).get('singing', [])
        speech_ts = aed_result.get('event2timestamps', {}).get('speech', [])
        music_ts = aed_result.get('event2timestamps', {}).get('music', [])
        print(f"✅ AED 结果:")
        print(f"  - singing: {len(singing_ts)} 段")
        print(f"  - speech: {len(speech_ts)} 段")
        print(f"  - music: {len(music_ts)} 段")
        if singing_ts:
            print(f"  - singing 前3段: {singing_ts[:3]}")
        if speech_ts:
            print(f"  - speech 前3段: {speech_ts[:3]}")
    else:
        print("❌ AED 返回空结果")
    
    # 3. 同时也测试VAD
    print("\n🔍 测试原始音频 VAD:")
    vad_mask, vad_fps = vad.get_voice_mask(tmp_wav)
    print(f"✅ VAD 人声占比: {vad_mask.mean():.1%}")
    
    # 清理
    os.remove(tmp_wav)
    
except Exception as e:
    import traceback
    print(f"❌ 错误: {e}")
    print(traceback.format_exc())

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

