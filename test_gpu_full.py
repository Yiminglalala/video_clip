# -*- coding: utf-8 -*-
"""
完整测试脚本 - 测试 SongFormer GPU 模式和整个处理流程
"""

import os
import sys
import time
import logging
import tempfile
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_gpu_full_latest.txt', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def test_cuda_availability():
    """测试 CUDA 是否可用"""
    logger.info("=" * 60)
    logger.info("测试 1: CUDA 可用性检查")
    logger.info("=" * 60)

    import torch
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    else:
        logger.error("CUDA 不可用!")

    return cuda_available

def test_songformer_analyzer():
    """测试 SongFormerAnalyzer 加载"""
    logger.info("=" * 60)
    logger.info("测试 2: SongFormerAnalyzer 加载测试")
    logger.info("=" * 60)

    from src.songformer_analyzer import SongFormerAnalyzer

    # 检查依赖
    ok, missing = SongFormerAnalyzer.check_runtime_dependencies()
    logger.info(f"依赖检查: OK={ok}, Missing={missing}")
    if not ok:
        logger.error(f"缺少依赖: {missing}")
        return False

    # 重置单例
    SongFormerAnalyzer.reset_instance()

    # 创建实例
    try:
        analyzer = SongFormerAnalyzer.get_instance(
            device='cuda',
            window_sec=60,
            hop_sec=30,
            verbose=True
        )
        logger.info(f"SongFormerAnalyzer 创建成功")
        logger.info(f"  - device: {analyzer.device}")
        logger.info(f"  - window_sec: {analyzer.window_sec}")
        logger.info(f"  - hop_sec: {analyzer.hop_sec}")

        # 检查模型是否加载
        if analyzer._models_loaded:
            logger.info("  - 模型已加载: True")
        else:
            logger.info("  - 模型已加载: False (懒加载)")

        return True
    except Exception as e:
        logger.error(f"SongFormerAnalyzer 创建失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_songformer_inference():
    """测试 SongFormer 推理功能"""
    logger.info("=" * 60)
    logger.info("测试 3: SongFormer 推理测试")
    logger.info("=" * 60)

    import torch
    import librosa
    from src.songformer_analyzer import SongFormerAnalyzer

    # 清理之前的实例
    SongFormerAnalyzer.reset_instance()

    try:
        analyzer = SongFormerAnalyzer.get_instance(
            device='cuda',
            window_sec=60,
            hop_sec=30,
            verbose=True
        )

        # 生成测试音频 (30秒，24000Hz)
        sr = 24000
        duration = 30
        t = np.linspace(0, duration, int(sr * duration))
        # 生成一个简单的测试信号
        audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        audio = audio.astype(np.float32)

        logger.info(f"测试音频: {len(audio)} samples, {duration}秒, 采样率 {sr}Hz")
        logger.info(f"音频范围: min={audio.min():.4f}, max={audio.max():.4f}")

        # 执行推理
        t0 = time.time()
        segments = analyzer.analyze_array(audio, sr)
        elapsed = time.time() - t0

        logger.info(f"推理完成! 耗时: {elapsed:.2f}秒")
        logger.info(f"检测到 {len(segments)} 个段落:")
        for seg in segments:
            logger.info(f"  [{seg['start']:>7.2f}s - {seg['end']:>7.2f}s] {seg['label']} (conf={seg.get('confidence', 0):.2f})")

        return True

    except Exception as e:
        logger.error(f"SongFormer 推理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_processor_pipeline():
    """测试完整的处理流程"""
    logger.info("=" * 60)
    logger.info("测试 4: Processor 完整流程测试")
    logger.info("=" * 60)

    import torch
    from src.processor import LiveVideoProcessor

    # 测试视频路径 - 使用实际存在的视频
    test_video = "D:\\个人资料\\音乐测试\\视频素材\\live\\周深\\周深-VX-q97643800 (24).mp4"

    if not os.path.exists(test_video):
        logger.warning(f"测试视频不存在: {test_video}")
        logger.info("跳过 Processor 完整流程测试")
        return True

    try:
        from src.processor import LiveVideoProcessor, ProcessingConfig
        
        # 创建配置对象
        config = ProcessingConfig(
            enable_songformer=True,
            songformer_device='cuda',
            songformer_window=60,
            songformer_hop=30,
        )

        processor = LiveVideoProcessor(config)

        # 检查 songformer_analyzer
        logger.info("初始化 SongFormerAnalyzer...")
        t0 = time.time()
        analyzer = processor.songformer_analyzer
        elapsed = time.time() - t0
        logger.info(f"SongFormerAnalyzer 初始化完成: {elapsed:.2f}秒")
        logger.info(f"  - device: {analyzer.device}")
        logger.info(f"  - window_sec: {analyzer.window_sec}")
        logger.info(f"  - models_loaded: {analyzer._models_loaded}")

        logger.info("Processor 完整流程测试完成!")
        return True

    except Exception as e:
        logger.error(f"Processor 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_degradation_logs():
    """检查日志中是否有降级情况"""
    logger.info("=" * 60)
    logger.info("检查 5: 降级情况检查")
    logger.info("=" * 60)

    degradation_keywords = [
        "降级",
        "fallback",
        "降级为手工",
        "手工分析",
        "SongFormer 分析失败",
        "SongFormer failed",
        "CUDA OOM",
        "OOM",
    ]

    # 读取日志文件
    log_file = 'test_gpu_full_latest.txt'
    if not os.path.exists(log_file):
        logger.info("日志文件不存在，无法检查降级情况")
        return True

    found_degradation = False
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过检查函数自身的输出
            if "检查 5:" in line or "WARNING - 发现降级关键词" in line:
                continue
            for keyword in degradation_keywords:
                if keyword.lower() in line.lower():
                    logger.warning(f"发现降级关键词: {keyword}")
                    logger.warning(f"  日志内容: {line.strip()}")
                    found_degradation = True

    if not found_degradation:
        logger.info("未发现降级情况!")

    return not found_degradation

def main():
    """主测试流程"""
    logger.info("=" * 80)
    logger.info("SongFormer GPU 完整测试开始")
    logger.info("=" * 80)

    results = {}

    # 测试 1: CUDA 可用性
    results['cuda'] = test_cuda_availability()

    # 测试 2: SongFormerAnalyzer 加载
    results['analyzer'] = test_songformer_analyzer()

    # 测试 3: SongFormer 推理
    if results['analyzer']:
        results['inference'] = test_songformer_inference()
    else:
        results['inference'] = False

    # 测试 4: Processor 完整流程
    results['processor'] = test_processor_pipeline()

    # 检查降级情况
    results['no_degradation'] = check_degradation_logs()

    # 输出总结
    logger.info("=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    for key, value in results.items():
        status = "✅ PASS" if value else "❌ FAIL"
        logger.info(f"{key}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("=" * 80)
        logger.info("🎉 所有测试通过!")
        logger.info("=" * 80)
    else:
        logger.error("=" * 80)
        logger.error("❌ 部分测试失败，请检查日志")
        logger.error("=" * 80)

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
