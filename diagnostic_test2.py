import os
import sys
import logging

sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/video_clip/output/diagnostic_test2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.processor import LiveVideoProcessor, ProcessingConfig

config = ProcessingConfig(
    output_dir=r"D:\video_clip\output\diagnostic_test2",
    enable_demucs=True,
    demucs_model="htdemucs_ft",
    enable_songformer=True,
    songformer_device="cuda",
    min_segment_duration=8.0,
    max_segment_duration=15.0,
    strict_songformer=True,
)

processor = LiveVideoProcessor(config)

input_video = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (10).mp4"
logger.info(f"开始诊断处理: {input_video}")

try:
    result, output_files = processor.process_video(input_video)
    logger.info(f"\n" + "="*80)
    logger.info(f"处理完成!")
    logger.info(f"Songs found: {len(result.songs)}")
    
    for i, song in enumerate(result.songs):
        logger.info(f"\n--- Song {i+1}: {song.song_name} ---")
        logger.info(f"  Start: {song.start_time:.1f}s, End: {song.end_time:.1f}s")
        logger.info(f"  Segments: {len(song.segments)}")
        
        logger.info(f"\n  [SongFormer原始标签]")
        for j, seg in enumerate(song.segments):
            orig_label = str(getattr(seg, 'original_label', getattr(seg, 'label', 'unknown')))
            current_label = str(getattr(seg, 'label', 'unknown'))
            logger.info(f"    [{j+1}] {orig_label} ({seg.start_time:.1f}-{seg.end_time:.1f}s) -> {current_label}")
        
        verse_count = 0
        chorus_count = 0
        talk_count = 0
        other_count = 0
        
        for j, seg in enumerate(song.segments):
            label = str(getattr(seg, 'label', 'unknown'))
            logger.info(f"    [{j+1}] {label} ({seg.start_time:.1f}-{seg.end_time:.1f}s)")
            
            if 'verse' in label.lower() or '主歌' in label:
                verse_count += 1
            elif 'chorus' in label.lower() or '副歌' in label:
                chorus_count += 1
            elif 'talk' in label.lower() or '讲话' in label:
                talk_count += 1
            else:
                other_count += 1
        
        logger.info(f"  统计: 主歌={verse_count}, 副歌={chorus_count}, 讲话={talk_count}, 其他={other_count}")
    
    logger.info(f"\n输出文件: {len(output_files)}个")
    for f in output_files:
        logger.info(f"  - {os.path.basename(f)}")
        
except Exception as e:
    import traceback
    logger.error(f"Error: {e}")
    logger.error(traceback.format_exc())