"""
多模态边界检测器 v9.0
整合 MERT、OCR、音频信号进行歌曲边界检测
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BoundaryDetector:
    """
    多模态边界检测器
    
    整合以下信号：
    1. MERT 结构边界预测
    2. OCR 标题画面检测
    3. 音频能量谷值
    4. 调性突变（Chroma Key）
    """
    
    def __init__(self, gpu_processor: 'GPUProcessor', config: 'GPUConfig'):
        self.gpu = gpu_processor
        self.config = config
        self.min_interval = config.MIN_SONG_INTERVAL
        self.min_duration = config.MIN_SONG_DURATION
    
    def detect(self,
              audio_path: str,
              vocal_path: Optional[str],
              video_path: Optional[str],
              total_duration: float) -> List[Tuple[float, float]]:
        """
        检测歌曲边界
        
        Args:
            audio_path: 原始音频路径
            vocal_path: Demucs 分离后人声路径
            video_path: 视频路径（用于 OCR）
            total_duration: 总时长
        
        Returns:
            [(start, end), ...] 每首歌的起止时间
        """
        boundaries = []
        
        # ═══════════════════════════════════════════════════════════════════════
        # 1. OCR 标题画面检测（最高优先级）
        # ═══════════════════════════════════════════════════════════════════════
        if video_path:
            ocr_boundaries = self._detect_ocr(video_path)
            boundaries.extend(ocr_boundaries)
            logger.info(f"OCR 检测到 {len(ocr_boundaries)} 个标题画面")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. MERT 结构边界预测
        # ═══════════════════════════════════════════════════════════════════════
        if self.config.ENABLE_MERT:
            mert_boundaries = self._detect_mert_boundaries(audio_path, total_duration)
            boundaries.extend(mert_boundaries)
            logger.info(f"MERT 检测到 {len(mert_boundaries)} 个结构边界")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. 音频能量谷值检测
        # ═══════════════════════════════════════════════════════════════════════
        audio_boundaries = self._detect_energy_valleys(audio_path, total_duration)
        boundaries.extend(audio_boundaries)
        logger.info(f"音频检测到 {len(audio_boundaries)} 个能量谷值")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4. 调性突变检测
        # ═══════════════════════════════════════════════════════════════════════
        chroma_boundaries = self._detect_chroma_changes(audio_path, total_duration)
        boundaries.extend(chroma_boundaries)
        logger.info(f"Chroma 检测到 {len(chroma_boundaries)} 个调性突变")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 5. 融合所有边界（AND 策略）
        # ═══════════════════════════════════════════════════════════════════════
        merged = self._merge_boundaries(boundaries, total_duration)
        
        logger.info(f"融合后共 {len(merged)} 首歌曲")
        return merged
    
    def _detect_ocr(self, video_path: str) -> List[float]:
        """OCR 标题画面检测"""
        try:
            from src.text_detector import TextFrameDetector
            
            detector = TextFrameDetector()
            boundaries = detector.find_title_boundaries(
                video_path,
                min_interval=self.min_interval,
                max_boundaries=20
            )
            return [float(t) for t in boundaries]
        except Exception as e:
            logger.warning(f"OCR 检测失败: {e}")
            return []
    
    def _detect_mert_boundaries(self, audio_path: str, total_duration: float) -> List[float]:
        """MERT 结构边界检测"""
        try:
            embeddings = self.gpu.extract_mert_embeddings(audio_path)
            if embeddings is None:
                return []
            
            # 简化：计算相邻帧嵌入的差异
            # 差异大的位置可能是结构边界
            diff = np.abs(np.diff(embeddings, axis=0))
            diff_magnitude = np.mean(diff, axis=1)
            
            # 平滑
            from scipy.ndimage import uniform_filter1d
            diff_smooth = uniform_filter1d(diff_magnitude.astype(float), size=10)
            
            # 找峰值
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(diff_smooth, distance=int(5 * 50))  # 至少间隔5秒
            
            # 转换为时间
            fps = 50
            boundary_times = [peaks[i] / fps for i in range(len(peaks))]
            
            return boundary_times
            
        except Exception as e:
            logger.warning(f"MERT 边界检测失败: {e}")
            return []
    
    def _detect_energy_valleys(self, audio_path: str, total_duration: float) -> List[float]:
        """音频能量谷值检测"""
        try:
            import librosa
            from scipy.signal import find_peaks
            
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # RMS 能量
            rms = librosa.feature.rms(y=y, hop_length=1024)[0]
            
            # 平滑
            from scipy.ndimage import uniform_filter1d
            fps = sr / 1024
            rms_smooth = uniform_filter1d(rms, size=int(fps * 3))
            
            # 找谷值（反向峰值）
            valleys, _ = find_peaks(-rms_smooth, distance=int(fps * self.min_interval))
            
            # 转换为时间
            boundary_times = [valleys[i] / fps for i in range(len(valleys))]
            
            return boundary_times
            
        except Exception as e:
            logger.warning(f"能量谷值检测失败: {e}")
            return []
    
    def _detect_chroma_changes(self, audio_path: str, total_duration: float) -> List[float]:
        """调性突变检测"""
        try:
            import librosa
            from scipy.signal import find_peaks
            
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Chroma 特征
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # 计算相邻帧的差异
            diff = np.abs(np.diff(chroma.T, axis=0))
            diff_magnitude = np.mean(diff, axis=1)
            
            # 平滑
            from scipy.ndimage import uniform_filter1d
            fps = sr / 512
            diff_smooth = uniform_filter1d(diff_magnitude.astype(float), size=int(fps * 2))
            
            # 找峰值
            peaks, _ = find_peaks(diff_smooth, distance=int(fps * self.min_interval))
            
            # 转换为时间
            boundary_times = [peaks[i] / fps for i in range(len(peaks))]
            
            return boundary_times
            
        except Exception as e:
            logger.warning(f"调性突变检测失败: {e}")
            return []
    
    def _merge_boundaries(self, 
                         boundaries: List[float], 
                         total_duration: float) -> List[Tuple[float, float]]:
        """
        合并边界，返回歌曲列表
        """
        if not boundaries:
            # 无边界，返回整首
            return [(0.0, total_duration)]
        
        # 去重 + 排序
        boundaries = sorted(set(boundaries))
        
        # 过滤太近的边界
        filtered = []
        last = -self.min_interval
        for b in boundaries:
            if b - last >= self.min_interval:
                filtered.append(b)
                last = b
        
        # 加上首尾
        filtered = [0.0] + filtered + [total_duration]
        
        # 过滤太短的歌曲
        songs = []
        for i in range(len(filtered) - 1):
            start = filtered[i]
            end = filtered[i + 1]
            if end - start >= self.min_duration:
                songs.append((start, end))
        
        # 至少一首
        if not songs:
            return [(0.0, total_duration)]
        
        return songs


def create_boundary_detector(gpu_processor: 'GPUProcessor', 
                           config: 'GPUConfig') -> BoundaryDetector:
    """工厂函数"""
    return BoundaryDetector(gpu_processor, config)
