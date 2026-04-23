"""
测试字幕烧录流程
"""
import sys
sys.path.insert(0, r"D:\video_clip")

import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 测试视频路径
VIDEO_PATH = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (1).mp4"

def main():
    logger.info(f"=" * 60)
    logger.info(f"测试视频: {VIDEO_PATH}")
    logger.info(f"视频存在: {os.path.exists(VIDEO_PATH)}")
    logger.info(f"=" * 60)

    if not os.path.exists(VIDEO_PATH):
        logger.error(f"视频文件不存在: {VIDEO_PATH}")
        return

    try:
        # 1. 导入模块
        logger.info("[Step 1] 导入模块...")
        from src.processor import LiveVideoProcessor, ProcessingConfig
        logger.info("模块导入成功")

        # 2. 创建配置
        logger.info("[Step 2] 创建配置...")
        config = ProcessingConfig(
            output_dir=r"D:\video_clip\output\test_subtitle",
            enable_subtitle=True,
            subtitle_mode="sentence",
            min_duration_limit=10.0,
            max_duration_limit=20.0,
        )
        logger.info(f"配置创建成功: enable_subtitle={config.enable_subtitle}")

        # 3. 创建处理器
        logger.info("[Step 3] 创建处理器...")
        processor = LiveVideoProcessor(config)
        logger.info("处理器创建成功")

        # 4. 运行分析
        logger.info("[Step 4] 开始分析视频...")
        def progress_callback(pct, msg):
            logger.info(f"[进度 {pct*100:.1f}%] {msg}")

        result, export_segments, subtitled_video_path = processor.analyze_video(
            VIDEO_PATH,
            singer="周深",
            progress_callback=progress_callback,
        )

        logger.info(f"=" * 60)
        logger.info(f"分析完成!")
        logger.info(f"歌曲数量: {len(result.songs)}")
        logger.info(f"导出片段数量: {len(export_segments)}")
        logger.info(f"字幕视频路径: {subtitled_video_path}")
        logger.info(f"字幕视频存在: {subtitled_video_path and os.path.exists(subtitled_video_path)}")
        logger.info(f"=" * 60)

        # 5. 检查缓存的ASR结果
        logger.info("[Step 5] 检查ASR缓存...")
        logger.info(f"_cached_asr_results 键数: {len(processor._cached_asr_results)}")
        for k, v in processor._cached_asr_results.items():
            sentences = v.get("sentences", [])
            error = v.get("error")
            logger.info(f"  歌曲{k}: {len(sentences)}句, error={error}")

        # 6. 检查字幕视频文件
        if subtitled_video_path and os.path.exists(subtitled_video_path):
            file_size = os.path.getsize(subtitled_video_path) / 1024 / 1024
            logger.info(f"[成功] 字幕视频已生成: {subtitled_video_path} ({file_size:.1f}MB)")
        else:
            logger.error(f"[失败] 字幕视频未生成!")

    except Exception as e:
        import traceback
        logger.error(f"测试失败: {e}")
        logger.error(f"详细错误:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()