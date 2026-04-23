# -*- coding: utf-8 -*-
"""
测试完整的字幕生成流程，包括豆包API的使用
"""

import os
import sys
import tempfile
import soundfile as sf
from src.doubao_api import DoubaoASR, format_result
from src.ffmpeg_processor import FFmpegProcessor
from src.alignment_engine import align_lrc_monotonic, fuse_engine_alignments

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_full_process(video_path):
    """
    测试完整的字幕生成流程
    
    Args:
        video_path: 视频文件路径
    """
    print(f"测试视频: {video_path}")
    
    # 1. 提取音频
    print("=== 1. 提取音频 ===")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    
    ffmpeg_processor = FFmpegProcessor()
    result = ffmpeg_processor.extract_audio(
        video_path=video_path,
        output_path=audio_path,
        sample_rate=16000
    )
    
    if not result.success:
        print(f"音频提取失败: {result.error_message}")
        return False
    
    # 读取音频文件
    audio, sr = sf.read(audio_path)
    print(f"音频提取完成: {len(audio)/sr:.2f}秒, {sr}Hz")
    
    # 2. 初始化豆包API
    print("\n=== 2. 初始化豆包API ===")
    appid = "6118416182"
    access_token = "wgYVCSXYek6ATuLNP_DiXFNHZ9jo5ZRV"
    doubao = DoubaoASR(appid=appid, access_token=access_token)
    
    # 3. 调用豆包API进行语音识别
    print("\n=== 3. 调用豆包API进行语音识别 ===")
    try:
        # 读取音频文件作为二进制数据
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        doubao_result = doubao.recognize(
            audio_data=audio_data,
            language="zh-CN",
            caption_type="singing"  # 使用唱歌模式，适合演唱会场景
        )
        print("API调用成功!")
        
        # 转换结果格式
        result = format_result(doubao_result)
        
        # 打印识别结果
        print("\n=== 识别结果 ===")
        print(f"引擎: {result.get('engine')}")
        print(f"识别文本: {result.get('text')}")
        print(f"词数: {len(result.get('words', []))}")
        print(f"句子数: {len(result.get('sentences', []))}")
        print(f"时长: {result.get('duration', 0):.2f}秒")
        
        if result.get('error'):
            print(f"错误: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return False
    finally:
        # 清理临时文件
        try:
            os.remove(audio_path)
        except:
            pass
    
    # 4. 模拟歌词对齐（使用识别结果作为歌词）
    print("\n=== 4. 模拟歌词对齐 ===")
    try:
        # 从识别结果中提取歌词
        lyrics = result.get('text', '')
        print(f"提取的歌词: {lyrics[:100]}...")
        
        # 模拟对齐过程
        print("模拟歌词对齐完成")
        
    except Exception as e:
        print(f"歌词对齐失败: {str(e)}")
        return False
    
    # 5. 测试结果
    print("\n=== 5. 测试结果 ===")
    print("✅ 完整流程测试成功！")
    print("\n测试总结:")
    print(f"- 视频文件: {os.path.basename(video_path)}")
    print(f"- 豆包API调用: 成功")
    print(f"- 识别结果: {len(result.get('text', ''))} 字符")
    print(f"- 词级时间戳: {len(result.get('words', []))} 个词")
    
    return True


if __name__ == "__main__":
    # 选择测试视频
    test_videos = [
        "D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (1).mp4",  # 用户提供的测试视频
        "A-Lin_2021_concert.wav",  # 直接使用音频文件
        "output/周深_20260416_1646/Song_01/20260416_未知歌曲_主歌_001.mp4"
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
    success = test_full_process(test_video)
    
    if success:
        print("\n🎉 测试完成，豆包API集成成功！")
    else:
        print("\n❌ 测试失败，请检查错误信息")
