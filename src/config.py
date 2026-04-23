"""
音频分析配置模块
集中管理所有硬编码阈值，便于调优和维护
"""

from dataclasses import dataclass
from typing import Final


@dataclass
class AudioConfig:
    """音频分析配置"""
    
    # ==================== 采样设置 ====================
    SAMPLE_RATE: Final[int] = 22050
    HOP_LENGTH: Final[int] = 1024
    
    # ==================== 段落切割 ====================
    MIN_SEGMENT_DURATION: Final[float] = 8.0     # 最小段落时长(秒)，降低以保留副歌/桥段等短段落
    MAX_SEGMENT_DURATION: Final[float] = 45.0     # 最大段落时长(秒)
    TARGET_SEGMENT_DURATION: Final[float] = 30.0 # 目标段落时长(秒)
    MIN_MERGE_DURATION: Final[float] = 12.0       # 合并固定时长(秒)
    
    # ==================== 能量检测 ====================
    AUDIO_ENERGY_THRESHOLD: Final[float] = 0.3    # 音频能量阈值
    
    # ==================== 歌曲检测 ====================
    MIN_SONG_INTERVAL: Final[float] = 15.0       # 最小歌曲间隔(秒)
    MIN_SONG_DURATION: Final[float] = 15.0       # 最小歌曲时长(秒)
    
    # ==================== VAD 设置 ====================
    VAD_THRESHOLD: Final[float] = 0.3            # Silero VAD 阈值 (降低以检测更多人声)
    VAD_MIN_SPEECH_MS: Final[int] = 300          # 最小语音时长(毫秒)
    VAD_MIN_SILENCE_MS: Final[int] = 500         # 最小静音时长(毫秒)
    
    # ==================== Demucs 设置 ====================
    DEMUCS_SEGMENT_SEC: Final[int] = 10          # Demucs 分段时长(秒)
    
    # ==================== SSM 设置 ====================
    SSM_SIMILARITY_THRESHOLD: Final[float] = 0.5       # 相似度阈值
    SSM_MIN_CHORUS_DURATION: Final[float] = 12.0       # 最小副歌时长(秒)
    SSM_MIN_BLOCK_DURATION: Final[float] = 15.0         # 最小重复块时长(秒)
    SSM_N_SAMPLES: Final[int] = 200                      # 下采样点数
    
    # ==================== 分类阈值 ====================
    # Stage 1 大类判断阈值
    SPEECH_SCORE_THRESHOLD: Final[float] = 6.0    # 讲话分类阈值
    TALK_SCORE_THRESHOLD: Final[float] = 6.0       # 互动分类阈值
    CROWD_SCORE_THRESHOLD: Final[float] = 3.5       # 掌声欢呼阈值 (演唱会优化：降低以识别更多观众合唱)
    AUDIENCE_SCORE_THRESHOLD: Final[float] = 4.0    # 观众合唱阈值 (演唱会优化：降低以识别更多观众合唱)
    
    # 静音检测
    SILENCE_RATIO_THRESHOLD: Final[float] = 0.7    # 静音比例阈值
    
    # ==================== Stage 2 音乐细分权重 ====================
    # 特征权重
    HAS_VOICE_WEIGHT: Final[float] = 2.5           # 人声权重
    VOICE_RATIO_LOW: Final[float] = 0.25           # 人声比例下限
    VOICE_RATIO_HIGH: Final[float] = 0.90           # 人声比例上限
    
    # 能量阈值
    ENERGY_THRESHOLD_RATIO: Final[float] = 1.10    # 能量阈值比例
    ENERGY_BONUS_MULTIPLIER: Final[float] = 22.0    # 能量奖励乘数
    ENERGY_BONUS_CAP: Final[float] = 9.0            # 能量奖励上限
    
    # beat regularity
    BEAT_REGULARITY_LOW: Final[float] = 0.15       # 低节拍规律性
    BEAT_REGULARITY_HIGH: Final[float] = 0.35       # 高节拍规律性
    
    # 位置上下文
    VERSE_POSITION_RATIO: Final[float] = 0.3       # 主歌位置比例
    CHORUS_POSITION_RATIO: Final[float] = 0.7       # 副歌位置比例
    
    # ==================== 歌曲内相对能量基准（优化1）====================
    CHORUS_RELATIVE_ENERGY_MIN: Final[float] = 0.80  # 副歌：歌曲内相对能量下限
    VERSE_RELATIVE_ENERGY_MAX: Final[float] = 1.20   # 主歌：歌曲内相对能量上限
    SONG_ENERGY_SMOOTH_SEC: Final[float] = 3.0       # 能量平滑窗口（秒）
    
    # ==================== Chroma Key 边界检测（优化4）====================
    CHROMA_KEY_CHANGE_THRESHOLD: Final[float] = 0.40  # 调性跳变余弦相似度骤降阈值
    CHROMA_KEY_WINDOW_SEC: Final[float] = 2.0         # Chroma 比较窗口（秒）
    BOUNDARY_AND_FUSION: Final[bool] = True            # 启用 AND 融合策略（替代加权和）
    
    # ==================== 频谱质心稳定性（优化3）====================
    CENTROID_STABILITY_WINDOW: Final[float] = 5.0     # 质心稳定性分析窗口（秒）
    CENTROID_GRADIENT_WINDOW: Final[float] = 2.0       # 质心梯度分析窗口（秒）
    
    # ==================== SSM 特征融合（优化2）====================
    SSM_FEATURE_WEIGHT: Final[float] = 5.0            # SSM 副歌似然特征权重
    SSM_CONFIDENCE_MIN: Final[float] = 0.25           # SSM 置信度下限


# 全局配置实例
CONFIG = AudioConfig()
