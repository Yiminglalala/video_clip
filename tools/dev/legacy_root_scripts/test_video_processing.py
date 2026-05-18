# -*- coding: utf-8 -*-
"""
完整视频处理测试 - 使用周深的视频进行测试
"""

import os
import sys
import time
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_video_processing.txt', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_full_video_processing():
    """测试完整的视频处理流程"""
    logger.info("=" * 80)
    logger.info("完整视频处理测试开始")
    logger.info("=" * 80)

    # 测试视频路径
    test_video = "D:\\个人资料\\音乐测试\\视频素材\\live\\周深\\周深-VX-q97643800 (24).mp4"

    if not os.path.exists(test_video):
        logger.error(f"测试视频不存在: {test_video}")
        return False

    logger.info(f"测试视频: {test_video}")

    try:
        from src.processor import LiveVideoProcessor, ProcessingConfig

        # 创建配置对象 - 启用 SongFormer GPU 模式
        config = ProcessingConfig(
            enable_songformer=True,
            songformer_device='cuda',
            songformer_window=60,
            songformer_hop=30,
            enable_subtitle=True,
            subtitle_model="small",
            subtitle_mode="sentence",
        )

        logger.info("创建 LiveVideoProcessor...")
        processor = LiveVideoProcessor(config)

        # 检查 songformer_analyzer
        logger.info("检查 SongFormerAnalyzer...")
        analyzer = processor.songformer_analyzer
        logger.info(f"  - device: {analyzer.device}")
        logger.info(f"  - window_sec: {analyzer.window_sec}")
        logger.info(f"  - models_loaded: {analyzer._models_loaded}")

        # 执行视频处理
        logger.info("开始视频处理...")
        t0 = time.time()

        # 定义进度回调
        def progress_callback(progress, message):
            logger.info(f"进度: {progress*100:.1f}% - {message}")

        result, output_files = processor.process_video(
            video_path=test_video,
            singer="周深",
            progress_callback=progress_callback,
        )

        elapsed = time.time() - t0

        logger.info("=" * 80)
        logger.info("视频处理完成!")
        logger.info(f"总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        logger.info(f"处理结果: {result}")
        logger.info("=" * 80)

        # 检查是否有降级
        if "降级" in str(result).lower() or "fallback" in str(result).lower():
            logger.error("检测到降级!")
            return False

        return True

    except Exception as e:
        logger.error(f"视频处理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    success = test_full_video_processing()

    logger.info("=" * 80)
    if success:
        logger.info("🎉 视频处理测试通过!")
    else:
        logger.error("❌ 视频处理测试失败")
    logger.info("=" * 80)

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
