# -*- coding: utf-8 -*-
"""
测试豆包语音API集成效果
"""

import os
import sys
import numpy as np
import soundfile as sf
from src.doubao_api import DoubaoASR, format_result
from src.ffmpeg_processor import FFmpegProcessor

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_doubao_api(media_path):
    """
    测试豆包API
    
    Args:
        media_path: 视频或音频文件路径
    """
    print(f"测试媒体: {media_path}")
    
    # 检查文件类型
    if media_path.endswith('.wav'):
        # 直接使用音频文件
        print("使用音频文件...")
        audio, sr = sf.read(media_path)
        print(f"音频加载完成: {len(audio)/sr:.2f}秒, {sr}Hz")
    else:
        # 提取音频
        print("提取音频...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            audio_path = tmp_audio.name
        
        processor = FFmpegProcessor()
        result = processor.extract_audio(
            video_path=media_path,
            output_path=audio_path,
            sample_rate=16000
        )
        
        if not result.success:
            print(f"音频提取失败: {result.error_message}")
            return None
        
        # 读取音频文件
        audio, sr = sf.read(audio_path)
        print(f"音频提取完成: {len(audio)/sr:.2f}秒, {sr}Hz")
        
        # 清理临时文件
        try:
            os.remove(audio_path)
        except:
            pass
    
    # 初始化豆包API
    print("初始化豆包API...")
    appid = "6118416182"
    access_token = "wgYVCSXYek6ATuLNP_DiXFNHZ9jo5ZRV"
    doubao = DoubaoASR(appid=appid, access_token=access_token)
    
    # 将音频转换为WAV格式
    print("准备音频数据...")
    import io
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="wav")
    audio_data = buffer.getvalue()
    
    # 调用API
    print("调用豆包API...")
    try:
        doubao_result = doubao.recognize(
            audio_data=audio_data,
            language="zh-CN",
            caption_type="auto"
        )
        print("API调用成功!")
        
        # 转换结果格式
        print("转换结果格式...")
        result = format_result(doubao_result)
        
        # 打印结果
        print("\n=== 识别结果 ===")
        print(f"引擎: {result.get('engine')}")
        print(f"识别文本: {result.get('text')}")
        print(f"词数: {len(result.get('words', []))}")
        print(f"句子数: {len(result.get('sentences', []))}")
        print(f"时长: {result.get('duration', 0):.2f}秒")
        
        if result.get('error'):
            print(f"错误: {result.get('error')}")
        
        # 打印词级时间戳
        print("\n=== 词级时间戳 ===")
        for i, word in enumerate(result.get('words', [])[:10]):  # 只显示前10个词
            print(f"{i+1}. {word.get('word')}: {word.get('start', 0):.2f}s - {word.get('end', 0):.2f}s")
        
        if len(result.get('words', [])) > 10:
            print(f"... 共 {len(result.get('words', []))} 个词")
        
        return result
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return None


if __name__ == "__main__":
    # 选择测试视频
    test_videos = [
        "A-Lin_2021_concert.wav",  # 直接使用音频文件
        "output/周深_20260416_1646/Song_01/20260416_未知歌曲_主歌_001.mp4",
        "output/A-Lin_20260415_1546/Song_01/20260415_《一夜》_主歌_002.mp4"
    ]
    
    # 选择第一个存在的视频
    test_video = None
    for video in test_videos:
        video_path = os.path.join(PROJECT_ROOT, video)
        if os.path.exists(video_path):
            test_video = video_path
            break
    
    if not test_video:
        print("未找到测试视频")
        sys.exit(1)
    
    # 运行测试
    test_doubao_api(test_video)
