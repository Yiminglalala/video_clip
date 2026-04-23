#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频字幕生成功能
"""

import os
import sys
import time
import tempfile

# 添加项目根目录到路径
PROJECT_ROOT = r"D:\video_clip"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 测试视频信息
VIDEO_PATH = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (6).mp4"
ARTIST = "周深"
SONG_TITLE = "大鱼"

# 导入核心函数
from app import extract_audio_from_video, run_pipeline_v2, preprocess_audio

# 测试函数
def test_video_subtitle():
    print("开始测试视频字幕生成功能...")
    print(f"视频路径: {VIDEO_PATH}")
    print(f"歌手: {ARTIST}")
    print(f"歌名: {SONG_TITLE}")
    print()
    
    # 确保输出目录存在
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 提取音频
    print("步骤1: 提取音频...")
    try:
        audio, sr = extract_audio_from_video(VIDEO_PATH, sr=16000)
        duration = len(audio) / sr
        print(f"✅ 音频提取完成 ({duration:.1f}秒, sr={sr})")
    except Exception as e:
        print(f"❌ 音频提取失败: {e}")
        return
    
    # 步骤2: 搜索歌词
    print("\n步骤2: 搜索歌词...")
    lyrics_text = ""
    try:
        from src.lyric_subtitle import SongMatch, fetch_lyrics
        song = SongMatch(title=SONG_TITLE, artist=ARTIST, provider="hint", confidence=1.0)
        payload = fetch_lyrics(song)
        if payload:
            lyrics_text = payload.get("lyrics", "").strip()
            print(f"✅ 歌词获取成功 ({payload.get('line_count', '?')}行)")
        else:
            print("❌ 未找到歌词")
            return
    except Exception as e:
        print(f"❌ 歌词搜索失败: {e}")
        return
    
    if not lyrics_text:
        print("❌ 歌词为空")
        return
    
    # 步骤3: 执行字幕生成
    print("\n步骤3: 生成字幕...")
    base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    timestamp = time.strftime("%H%M%S")
    out_name = f"{base_name}_{timestamp}_subtitled.mp4"
    output_path = os.path.join(output_dir, out_name)
    print(f"输出文件: {output_path}")
    
    # 定义日志函数
    def add_log(message):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {message}")
    
    # 模拟 session_state
    class MockSessionState:
        def __init__(self):
            self.output_dir = output_dir
            self._log_func = add_log
            self.preprocessing = True
            self.whisper_model_name = "medium"
            self.matching_strictness = "standard"
            self.low_conf_threshold = 0.42
            self.max_drift_sec = 1.2
            self.enable_quality_gate = True
            self.lyrics_title_hint = SONG_TITLE
            self.lyrics_artist_hint = ARTIST
            self.lyrics_offset_sec = 0.0
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # 模拟 Streamlit
    class MockStreamlit:
        def __init__(self):
            self.session_state = MockSessionState()
        
        def spinner(self, text):
            class MockSpinner:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockSpinner()
        
        def success(self, message):
            print(f"✅ {message}")
        
        def error(self, message):
            print(f"❌ {message}")
        
        def warning(self, message):
            print(f"⚠️ {message}")
        
        def info(self, message):
            print(f"ℹ️ {message}")
        
        def caption(self, message):
            print(f"📝 {message}")
        
        def dataframe(self, data, **kwargs):
            pass
        
        def video(self, path):
            pass
        
        def download_button(self, *args, **kwargs):
            pass
        
        def expander(self, *args, **kwargs):
            class MockExpander:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockExpander()
        
        def metric(self, *args):
            pass
        
        def columns(self, *args):
            class MockColumn:
                def metric(self, *args):
                    pass
            return [MockColumn() for _ in args]
        
        def checkbox(self, *args, **kwargs):
            return True
        
        def text_input(self, *args, **kwargs):
            return kwargs.get('value', '')
        
        def selectbox(self, *args, **kwargs):
            return kwargs.get('value', '')
        
        def slider(self, *args, **kwargs):
            return kwargs.get('value', 0)
        
        def number_input(self, *args, **kwargs):
            return kwargs.get('value', 0)
        
        def button(self, *args, **kwargs):
            return False
        
        def file_uploader(self, *args, **kwargs):
            return None
        
        def toast(self, message):
            print(f"🍞 {message}")
    
    # 替换 st
    import app
    app.st = MockStreamlit()
    
    try:
        # 预处理音频
        needs_preprocessing = False  # Fusion 模式不需要预处理
        if needs_preprocessing:
            add_log("音频预处理...")
            audio = preprocess_audio(audio, sr=sr)
            add_log("预处理完成")
        
        # 执行流水线
        success, msg_or_data = run_pipeline_v2(
            video_path=VIDEO_PATH,
            audio_16k=audio,
            sr=sr,
            lrc_text=lyrics_text,
            output_path=output_path,
            engine="fusion",
            strictness="standard",
            low_conf_threshold=0.42,
            max_drift_sec=1.2,
            title_hint=SONG_TITLE,
            artist_hint=ARTIST,
        )
        
        if success:
            data = msg_or_data
            print("\n✅ 处理成功！")
            print(f"输出路径: {output_path}")
            print(f"匹配行数: {data['match_count']}")
            print(f"Interpolated行数: {data.get('interpolated_count', 0)}")
            print(f"Review行数: {data.get('review_count', 0)}")
            print(f"总导出行数: {len(data['matches'])}")
        else:
            print(f"\n❌ 处理失败: {msg_or_data}")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_subtitle()
