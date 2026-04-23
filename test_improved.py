# -*- coding: utf-8 -*-
"""
改进验证测试：方案 C + A 已实施，验证分类效果
"""
import sys
import os
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.processor import LiveVideoProcessor, ProcessingConfig

VIDEO_PATH = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (24).mp4"
OUTPUT_DIR = r"D:\video_clip\test_output_improved"

def main():
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"视频文件不存在: {VIDEO_PATH}")
        return

    config = ProcessingConfig(
        output_dir=OUTPUT_DIR,
        min_segment_duration=8.0,
        max_segment_duration=15.0,
        enable_songformer=True,
        strict_songformer=True,
        enable_demucs=False,
    )

    processor = LiveVideoProcessor(config)

    def on_progress(progress, msg=""):
        pct = int(progress * 100)
        logger.info(f"[{pct:3d}%] {msg}")

    t0 = time.time()
    result, output_files = processor.process_video(
        VIDEO_PATH,
        singer="周深",
        concert="",
        progress_callback=on_progress,
    )
    elapsed = time.time() - t0

    logger.info("=" * 50)
    logger.info(f"切片完成！共 {len(result.songs)} 首歌，{len(output_files)} 个片段，耗时 {elapsed:.1f}s")

    for i, song in enumerate(result.songs):
        seg_types = {}
        for seg in song.segments:
            lt = str(seg.label)
            seg_types[lt] = seg_types.get(lt, 0) + 1
        type_str = " / ".join(f"{k}({v})" for k, v in seg_types.items())
        logger.info(f"  歌曲{i+1}: [{song.start_time:.1f}-{song.end_time:.1f}s] "
                     f"{song.song_name or song.song_title or '未知'} | {type_str}")

    for f in output_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        logger.info(f"  -> {os.path.basename(f)} ({size_mb:.1f}MB)")

if __name__ == "__main__":
    main()
