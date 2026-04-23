#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试字幕生成功能
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from app import run_pipeline_v2
import numpy as np
import librosa

def test_subtitle_generation():
    """测试字幕生成功能"""
    print("开始测试字幕生成功能...")
    
    # 测试视频路径
    video_path = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (6).mp4"
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return False
    
    # 提取音频
    print("提取音频...")
    import subprocess
    import tempfile
    
    temp_audio = tempfile.mktemp(suffix=".wav")
    
    try:
        # 使用ffmpeg提取音频
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            temp_audio
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"提取音频失败: {result.stderr}")
            return False
        
        print("音频提取成功")
        
        # 加载音频
        print("加载音频...")
        y, sr = librosa.load(temp_audio, sr=16000, mono=True)
        
        # 歌词文本（周深 - 大鱼）
        lrc_text = """
[00:00.00] 作词：尹约
[00:01.00] 作曲：钱雷
[00:02.00] 演唱：周深
[00:03.00]
[00:11.00] 海浪无声将夜幕深深淹没
[00:16.00] 漫过天空尽头的角落
[00:22.00] 大鱼在梦境的缝隙里游过
[00:27.00] 凝望你沉睡的轮廓
[00:33.00] 看海天一色 听风起雨落
[00:38.00] 执子手 吹散苍茫茫烟波
[00:44.00] 大鱼的翅膀 已经太辽阔
[00:49.00] 我松开时间的绳索
[00:55.00] 怕你飞远去 怕你离我而去
[01:00.00] 更怕你永远停留在这里
[01:06.00] 每一滴泪水 都向你流淌去
[01:11.00] 倒流回最初的相遇
[01:17.00]
[01:35.00] 海浪无声将夜幕深深淹没
[01:40.00] 漫过天空尽头的角落
[01:46.00] 大鱼在梦境的缝隙里游过
[01:51.00] 凝望你沉睡的轮廓
[01:57.00] 看海天一色 听风起雨落
[02:02.00] 执子手 吹散苍茫茫烟波
[02:08.00] 大鱼的翅膀 已经太辽阔
[02:13.00] 我松开时间的绳索
[02:19.00] 怕你飞远去 怕你离我而去
[02:24.00] 更怕你永远停留在这里
[02:30.00] 每一滴泪水 都向你流淌去
[02:35.00] 倒流回最初的相遇
[02:41.00]
[03:01.00] 怕你飞远去 怕你离我而去
[03:06.00] 更怕你永远停留在这里
[03:12.00] 每一滴泪水 都向你流淌去
[03:17.00] 倒流回最初的相遇
[03:23.00]
"""
        
        # 输出路径
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / "test_subtitle_output.mp4")
        
        # 运行字幕生成
        print("开始生成字幕...")
        start_time = time.time()
        
        success, result = run_pipeline_v2(
            video_path=video_path,
            audio_16k=y,
            sr=sr,
            lrc_text=lrc_text,
            output_path=output_path,
            engine="fusion",
            strictness="loose",
            low_conf_threshold=0.25,
            max_drift_sec=2.5,
            title_hint="大鱼",
            artist_hint="周深"
        )
        
        end_time = time.time()
        print(f"处理时间: {end_time - start_time:.2f}秒")
        
        if success:
            print("字幕生成成功！")
            print(f"输出文件: {output_path}")
            return True
        else:
            print(f"字幕生成失败: {result}")
            return False
            
    finally:
        # 清理临时文件
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    success = test_subtitle_generation()
    if success:
        print("测试成功！")
        sys.exit(0)
    else:
        print("测试失败！")
        sys.exit(1)