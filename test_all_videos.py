import logging
import sys
import os
sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.processor import LiveVideoProcessor, ProcessingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

config = ProcessingConfig(
    output_dir=r"D:\video_clip\output\test_all_videos",
    enable_demucs=True,
    demucs_model="htdemucs_ft",
    enable_songformer=True,
    songformer_device="cuda",
    min_segment_duration=8.0,
    max_segment_duration=15.0,
    strict_songformer=True,
    enable_subtitle=False,  # 禁用字幕功能
)

processor = LiveVideoProcessor(config)

# 禁用歌名识别和歌词识别
processor._disable_song_identify = True
processor._disable_lyrics_identify = True

# 四个测试视频
videos = [
    r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (6).mp4",  # 1. 全部都是唱歌，没有讲话
    r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (17).mp4", # 2. 0-20唱歌，21-44乐器solo，45-56合唱
    r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (20).mp4",  # 3. 前30秒唱歌，55秒后一直到最后歌手讲话
    r"E:\BaiduNetdiskDownload\video_20260412_211404.mp4",  # 4. 0-23秒歌手讲话，剩余唱歌
]

for i, video_path in enumerate(videos, 1):
    logger.info(f"\n{'='*80}")
    logger.info(f"开始测试视频 {i}/4: {video_path}")
    logger.info(f"{'='*80}")
    
    try:
        result, output_files = processor.process_video(video_path)
        logger.info(f"视频 {i} 处理完成!")
        
        # 输出详细结果
        logger.info(f"\nSongs found: {len(result.songs)}")
        
        for j, song in enumerate(result.songs):
            logger.info(f"\n--- Song {j+1}: {song.song_name} ---")
            logger.info(f"  Start: {song.start_time:.1f}s, End: {song.end_time:.1f}s")
            logger.info(f"  Segments: {len(song.segments)}")
            
            logger.info(f"\n  [SongFormer原始标签]")
            for k, seg in enumerate(song.segments):
                orig_label = str(getattr(seg, 'original_label', getattr(seg, 'label', 'unknown')))
                current_label = str(getattr(seg, 'label', 'unknown'))
                logger.info(f"    [{k+1}] {orig_label} ({seg.start_time:.1f}-{seg.end_time:.1f}s) -&gt; {current_label}")
            
            verse_count = 0
            chorus_count = 0
            talk_count = 0
            other_count = 0
            
            logger.info(f"\n  [最终标签]")
            for k, seg in enumerate(song.segments):
                label = str(getattr(seg, 'label', 'unknown'))
                logger.info(f"    [{k+1}] {label} ({seg.start_time:.1f}-{seg.end_time:.1f}s)")
                
                if 'verse' in label.lower() or '主歌' in label:
                    verse_count += 1
                elif 'chorus' in label.lower() or '副歌' in label:
                    chorus_count += 1
                elif 'talk' in label.lower() or '讲话' in label:
                    talk_count += 1
                else:
                    other_count += 1
            
            logger.info(f"  统计: 主歌={verse_count}, 副歌={chorus_count}, 讲话={talk_count}, 其他={other_count}")
        
    except Exception as e:
        logger.error(f"视频 {i} 处理失败: {e}")
        import traceback
        traceback.print_exc()

logger.info(f"\n{'='*80}")
logger.info("所有视频测试完成!")
logger.info(f"{'='*80}")