import os
import sys
import logging

# 设置日志级别为DEBUG，确保能看到[REFINE-DEBUG]的输出
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.processor import LiveVideoProcessor, ProcessingConfig

# 配置
config = ProcessingConfig(
    output_dir=r"D:\video_clip\output\test_only_video2",
    enable_demucs=True,
    demucs_model="htdemucs_ft",
    enable_songformer=True,
    songformer_device="cuda",
    min_segment_duration=8.0,
    max_segment_duration=15.0,
    strict_songformer=True,
    enable_subtitle=False,
)

processor = LiveVideoProcessor(config)

# 禁用歌名识别和歌词识别
processor._disable_song_identify = True
processor._disable_lyrics_identify = True

# 视频2路径
video2 = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (17).mp4"

print("\n" + "="*80)
print(f"开始测试视频2: {video2}")
print("="*80)

try:
    result, output_files = processor.process_video(video2)
    print("\n" + "="*80)
    print("视频2处理完成!")
    print("="*80)
    
    if result and result.songs:
        print(f"\nSongs found: {len(result.songs)}")
        for i, song in enumerate(result.songs):
            print(f"\n--- Song {i+1}: {song.song_name} ---")
            print(f"  Start: {song.start_time}s, End: {song.end_time}s")
            print(f"  Segments: {len(song.segments)}")
            print("\n  [最终标签]")
            for idx, seg in enumerate(song.segments):
                print(f"    [{idx+1}] {seg.label} ({seg.start_time}-{seg.end_time}s) (conf={seg.confidence:.2f})")
                
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    print(traceback.format_exc())

print("\n" + "="*80)
print("测试完成!")
print("="*80)