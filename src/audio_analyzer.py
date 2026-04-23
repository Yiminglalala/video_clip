"""
Live视频切片系统 - 音频分析和段落识别模块 v4.0
优化要点：
  A. Demucs 人声分离：先把人声轨和伴奏轨分离，
       - 人声轨 -> VAD / 歌手讲话检测（更纯净）
       - 全轨   -> 能量/边界检测（保留鼓点等信息）
       - 无 GPU / 未安装 demucs 时自动降级，不影响原有流程
  D. 动态归一化：分析完整音频后计算全局特征统计量
       （rms_p10/p50/p90、zcr_mean、centroid_mean 等），
       所有分类阈值改为"相对全局"的百分位比较，
       避免不同音量/录音质量的视频得到截然不同的阈值行为
  E. SSM 结构分析（新增）：基于自相似矩阵识别副歌重复模式
       - 副歌在歌曲中会出现多次，且旋律/和声相似
       - 通过计算音频帧之间的 MFCC+Chroma 相似度构建 SSM
       - SSM 中的对角线平行块 -> 重复段落（副歌候选）
       - 与特征分类融合，提升副歌识别准确率
  原有功能：
  - 歌曲边界：掌声检测 + 能量包络谷值 + Chroma跳变
  - 段落分类：6类（前奏/主歌/副歌/观众合唱/歌手讲话/尾奏）+ 多维特征
  - VAD：优先使用 Silero VAD，降级到 librosa 规则
"""

import numpy as np
import os
import json
import warnings
import logging
import time
import tempfile
import shutil
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple, Any
import librosa
import scipy.signal as signal
import scipy.ndimage

# 导入配置模块
from .config import CONFIG

logger = logging.getLogger(__name__)

# ── 可选依赖 ──────────────────────────────────────────────────────────────────

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Pedalboard 音频处理（降噪/增强）
try:
    import pedalboard
    PEDALBOARD_AVAILABLE = True
    logger.info("Pedalboard 可用")
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logger.info("Pedalboard 未安装")

# ── Demucs 人声分离 ───────────────────────────────────────────────────────────

DEMUCS_AVAILABLE = False

def _try_load_demucs() -> bool:
    """检测 demucs 是否可用（不预加载模型，节省内存）"""
    global DEMUCS_AVAILABLE
    if DEMUCS_AVAILABLE:
        return True
    try:
        import demucs  # noqa: F401
        DEMUCS_AVAILABLE = True
        logger.info("Demucs 可用")
        return True
    except ImportError:
        logger.info("Demucs 未安装，人声分离功能将跳过（pip install demucs）")
        return False

# 启动时探测一次
_try_load_demucs()


def separate_vocals(
    audio_path: str,
    out_dir: Optional[str] = None,
    model: str = "htdemucs_ft",
    device: str = "auto",
) -> Tuple[Optional[str], Optional[str]]:
    """
    使用 Demucs 分离人声和伴奏。

    Args:
        audio_path: 输入音频路径（WAV/MP3）
        out_dir: 输出目录（None 则使用系统临时目录）
        model: Demucs 模型名称，'htdemucs_ft' 质量最高（GPU推荐），'htdemucs' 次之，'mdx' 最快
        device: 'auto'='cuda' if GPU else 'cpu'（P0: 有5060Ti自动用GPU）

    Returns:
        (vocals_path, no_vocals_path) 或 (None, None) 失败时
    """
    try:
        import demucs  # noqa: F401
    except ImportError:
        logger.info("Demucs 未安装")
        return None, None

    try:
        import torch
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        if out_dir is None:
            out_dir = tempfile.mkdtemp(prefix="demucs_")

        # 选择设备（P0: 优先用GPU）
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Demucs 分离: {os.path.basename(audio_path)}，设备={device}，模型={model}")

        # 加载模型（GPU模式）
        demucs_model = get_model(model)
        demucs_model.to(device)
        demucs_model.eval()

        # 加载音频（统一到模型采样率）
        wav, sr = sf.read(audio_path, dtype='float32')
        sr_out = sr
        if sr != demucs_model.samplerate:
            import librosa
            wav = librosa.resample(wav.T, orig_sr=sr, target_sr=demucs_model.samplerate).T
            sr_out = demucs_model.samplerate
        wav = torch.from_numpy(wav).float().T
        sr = sr_out

        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        target_length = wav.shape[1]
        if target_length % 16 != 0:
            padded_length = ((target_length // 16) + 1) * 16
            wav = torch.nn.functional.pad(wav, (0, padded_length - target_length))
            logger.info(f"Demucs 填充音频: {target_length} -> {padded_length}")

        wav = wav.unsqueeze(0).to(device)  # [1, 2, T]

        # 推理（手动分块处理，避免超过模型训练长度）
        # htdemucs 训练长度约 343980 样本 @ 44100Hz ≈ 7.8 秒
        # 使用 segment 参数控制每块长度（秒）
        segment_sec = 7  # 每块 7 秒，略小于训练长度
        segment_samples = int(segment_sec * demucs_model.samplerate)

        # 确保 segment_samples 能被 16 整除
        segment_samples = (segment_samples // 16) * 16

        with torch.no_grad():
            out = apply_model(
                demucs_model, wav,
                device=device,
                shifts=1,
                split=True,
                overlap=0.1,
                progress=False,
                segment=segment_sec,
            )

        if isinstance(out, torch.Tensor):
            sources = out
        elif isinstance(out, (list, tuple)) and len(out) >= 1:
            sources = out[0]
            if isinstance(sources, (list, tuple)):
                sources = sources[0]
        else:
            raise RuntimeError(f"Demucs apply_model 返回了意外结构: {type(out)}")

        logger.info(f"[DEMUCS-DIAG] out type={type(out)}, len={len(out)}")
        if hasattr(out, 'shape'):
            logger.info(f"[DEMUCS-DIAG] out.shape={out.shape}")
        sources = sources[0]
        source_names = demucs_model.sources  # e.g. ['drums','bass','other','vocals']

        if 'vocals' not in source_names:
            logger.warning(f"模型 {model} 不含 vocals 轨，跳过分离")
            return None, None

        vocals_idx = source_names.index('vocals')
        vocals_wav = sources[vocals_idx]  # [2, T]

        # no_vocals = 其余轨道之和
        no_vocals_wav = sum(
            sources[i] for i in range(len(source_names)) if i != vocals_idx
        )

        # 保存采样率（在清理模型之前）
        sr_out = demucs_model.samplerate

        # 显存清理（释放 Demucs 模型内存）
        del demucs_model, sources, wav
        if device == "cuda":
            torch.cuda.empty_cache()

        def _save(wav_tensor, name):
            path = os.path.join(out_dir, name)
            mono = wav_tensor.mean(dim=0, keepdim=True).cpu().numpy()
            peak = np.abs(mono).max()
            if peak > 0:
                mono = mono / peak * 0.95
            sf.write(path, mono.T, sr_out)
            return path

        vocals_path    = _save(vocals_wav,    "vocals.wav")
        no_vocals_path = _save(no_vocals_wav, "no_vocals.wav")

        logger.info(f"Demucs 分离完成: vocals={vocals_path}")
        return vocals_path, no_vocals_path

    except Exception as e:
        logger.warning(f"Demucs 分离失败，降级到原始音频: {e}")
        return None, None


# ── Silero VAD（语音活动检测）────────────────────────────────────────────────

SILERO_AVAILABLE = False
_silero_model = None
_silero_utils = None
_silero_permanently_unavailable = False
_silero_unavailable_reason = ""

def _try_load_silero():
    """懒加载 Silero VAD 模型（带重试和 HuggingFace 镜像）"""
    global SILERO_AVAILABLE, _silero_model, _silero_utils
    global _silero_permanently_unavailable, _silero_unavailable_reason
    if _silero_model is not None:
        return True
    if _silero_permanently_unavailable:
        logger.warning(
            "Silero VAD 已禁用（依赖不满足），直接降级 librosa。原因: %s",
            _silero_unavailable_reason or "unknown",
        )
        return False
    if not TORCH_AVAILABLE:
        logger.warning("Torch 不可用，Silero VAD 无法加载")
        return False

    # torchaudio>=2.9 在 Windows 上默认走 torchcodec 音频 I/O，容易因运行时依赖导致加载失败。
    # 默认直接降级 librosa，避免每次推理阶段触发超长异常与卡顿；如需强制尝试，可设置 SILERO_FORCE_TORCHAUDIO_CODEC=1。
    try:
        import torchaudio
        import re
        ta_version = str(getattr(torchaudio, "__version__", "") or "")
        m = re.match(r"^(\d+)\.(\d+)", ta_version)
        ta_major = int(m.group(1)) if m else 0
        ta_minor = int(m.group(2)) if m else 0
        force_codec = os.environ.get("SILERO_FORCE_TORCHAUDIO_CODEC", "").strip() == "1"
        if (ta_major, ta_minor) >= (2, 9) and not force_codec:
            _silero_permanently_unavailable = True
            _silero_unavailable_reason = (
                f"torchaudio={ta_version} uses torchcodec on Windows; disabled by default"
            )
            logger.warning("Silero VAD 已禁用：%s", _silero_unavailable_reason)
            return False
    except Exception:
        # 不阻断主流程，继续走后续加载逻辑
        pass
    
    # 设置 HuggingFace 国内镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    
    for attempt in range(3):
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                # 避免重试时反复触发远程下载；失败后按异常类型决定是否继续重试
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            _silero_model = model
            _silero_utils = utils
            SILERO_AVAILABLE = True
            logger.info(f"Silero VAD 加载成功 (attempt {attempt+1})")
            return True
        except Exception as e:
            err = str(e)
            err_l = err.lower()
            logger.warning(f"Silero VAD 加载失败 (attempt {attempt+1}/3): {err}")
            # 依赖缺失/不兼容时无需重试，直接禁用并走 librosa，避免卡住
            hard_fail_markers = (
                "no module named 'torchaudio'",
                "requires torchcodec",
                "could not load libtorchcodec",
                "torchcodec",
            )
            if any(m in err_l for m in hard_fail_markers):
                _silero_permanently_unavailable = True
                _silero_unavailable_reason = err[:300]
                logger.warning("Silero VAD 依赖不满足，已永久禁用本进程内重试。")
                return False
            if attempt < 2:
                import time; time.sleep(1)
    
    logger.warning("Silero VAD 加载最终失败，将使用 librosa 规则 VAD（功能不受影响，精度略低）")
    return False


# ── FireRedVAD（语音活动检测 + 音频事件检测）─────────────────────────────────

FIREREDVAD_AVAILABLE = False
_firered_vad_model = None
_firered_aed_model = None
_FIREDRED_VAD_MODEL_DIR = "D:/video_clip/pretrained_models/xukaituo/FireRedVAD/VAD"
_FIREDRED_AED_MODEL_DIR = "D:/video_clip/pretrained_models/xukaituo/FireRedVAD/AED"


def _try_load_firered_vad() -> bool:
    """懒加载 FireRedVAD + AED 模型"""
    global FIREREDVAD_AVAILABLE, _firered_vad_model, _firered_aed_model
    if _firered_vad_model is not None:
        return True

    try:
        from fireredvad import FireRedVad, FireRedVadConfig, FireRedAed, FireRedAedConfig

        _firered_vad_model = FireRedVad.from_pretrained(
            _FIREDRED_VAD_MODEL_DIR,
            FireRedVadConfig(use_gpu=True)
        )
        _firered_aed_model = FireRedAed.from_pretrained(
            _FIREDRED_AED_MODEL_DIR,
            FireRedAedConfig(
                use_gpu=True,
                speech_threshold=0.05,
                singing_threshold=0.1,
                music_threshold=0.1,
                smooth_window_size=3,
                min_event_frame=5,
            )
        )
        FIREREDVAD_AVAILABLE = True
        logger.info("FireRedVAD + AED 模型加载成功")
        return True
    except Exception as e:
        logger.warning(f"FireRedVAD 加载失败: {e}，将降级到其他 VAD")
        return False


try:
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False

# ── 数据结构 ──────────────────────────────────────────────────────────────────

# ── 段落标签常量 v6.0（层级分类体系）─────────────────────────────────────
# 第一层：大类
LABEL_SILENCE  = 'silence'    # 静默（过渡用，不输出切片）
LABEL_SPEECH   = 'speech'     # 纯讲话（无音乐伴奏）
LABEL_TALK     = 'talk'       # 暖场互动（半说半唱，有低音量伴奏）
LABEL_CROWD    = 'crowd'      # 纯掌声/欢呼/尖叫（无旋律）
LABEL_AUDIENCE = 'audience'   # 观众大合唱（有旋律跟着唱）
# 第二层：音乐细分
LABEL_INTRO      = 'intro'      # 前奏（歌曲开头纯乐器）
LABEL_INTERLUDE = 'interlude'  # 间奏（歌曲中间纯乐器）
LABEL_SOLO      = 'solo'       # 乐器独奏
LABEL_VERSE     = 'verse'      # 主歌
LABEL_CHORUS    = 'chorus'     # 副歌/高潮
LABEL_OUTRO     = 'outro'      # 尾奏
LABEL_OTHER     = 'other'      # 其他

# 所有可能的标签
ALL_LABELS = [
    LABEL_SILENCE, LABEL_SPEECH, LABEL_TALK, LABEL_CROWD, LABEL_AUDIENCE,
    LABEL_INTRO, LABEL_INTERLUDE, LABEL_SOLO, LABEL_VERSE, LABEL_CHORUS,
    LABEL_OUTRO, LABEL_OTHER,
]

# 标签层级（用于两阶段分类）
LAYER1_LABELS = {LABEL_SILENCE, LABEL_SPEECH, LABEL_TALK, LABEL_CROWD}  # 非音乐大类
LAYER2_LABELS = {LABEL_INTRO, LABEL_INTERLUDE, LABEL_SOLO, LABEL_VERSE, LABEL_CHORUS, LABEL_OUTRO}  # 音乐细分
# audience 比较特殊：既有音乐成分也有观众成分，在 Stage1 中检测

# 中文映射
LABEL_CN = {
    # 传统分类器标签
    'silence':  '静默',
    'speech':   '讲话串场',
    'talk':     '讲话串场',
    'crowd':    '合唱',
    'audience': '合唱',
    'intro':    '前奏',
    'interlude': '间奏',
    'solo':     '独奏',
    'verse':    '主歌',
    'chorus':   '副歌',
    'outro':    '尾奏',
    'other':    '其他',
    # SongFormer 原始标签
    'prechorus': '导歌',
    'bridge':   '桥段',
    'inst':     '纯音乐',
}

# CSS 颜色映射（用于 UI 显示）
LABEL_COLORS = {
    LABEL_SILENCE:  '#B0BEC5',
    LABEL_SPEECH:   '#78909C',
    LABEL_TALK:     '#90A4AE',
    LABEL_CROWD:    '#CE93D8',
    LABEL_AUDIENCE: '#AB47BC',
    LABEL_INTRO:    '#42A5F5',
    LABEL_INTERLUDE: '#29B6F6',
    LABEL_SOLO:     '#26C6DA',
    LABEL_VERSE:    '#66BB6A',
    LABEL_CHORUS:   '#FF7043',
    LABEL_OUTRO:    '#FFA726',
    LABEL_OTHER:    '#BDBDBD',
}

# ── 算法参数常量 ────────────────────────────────────────────────────────────────

# SSM（自相似矩阵）参数
SSM_SIMILARITY_THRESHOLD = 0.5      # 相似度阈值，0.5表示中等相似，经验值
SSM_PERCENTILE_THRESHOLD = 70       # 动态阈值百分位，70%用于去除噪声
SSM_MIN_CHORUS_DURATION = 12.0      # 最小副歌时长（秒），基于音乐结构统计
SSM_MIN_BLOCK_DURATION = 15.0       # 最小块时长（秒）
SSM_STEP_SIZE_RATIO = 2             # 步长比例（block_size // STEP_SIZE_RATIO）

# 音频分析参数
AUDIO_DEFAULT_SR = 22050              # 默认采样率
AUDIO_DEFAULT_HOP_LENGTH = 512        # 默认帧移
AUDIO_MIN_SEGMENT_DURATION = 8.0      # 最小段落时长（秒）
AUDIO_ENERGY_THRESHOLD = 0.3          # 能量阈值


# ═══════════════════════════════════════════════════════════════════════════
# 常量定义（优化：消除魔数）
# ═══════════════════════════════════════════════════════════════════════════

# 段落切割
MIN_SEGMENT_DURATION = 25.0    # 最小段落时长（秒）
MAX_SEGMENT_DURATION = 45.0    # 最大段落时长（秒）
TARGET_SEGMENT_DURATION = 30.0 # 目标段落时长（秒）

# 节拍对齐
BEAT_ALIGN_TOLERANCE = 2.0     # 节拍对齐容差（秒）

# 段落后处理
MIN_MERGE_DURATION = 12.0      # 合并过短段落的阈值
MIN_SEGMENT_FOR_POSTPROC = 3.0 # 有效段落的最小时长

# SSM 分析
SSM_MIN_BLOCK_DURATION = 15.0  # 最小重复块时长（秒）
SSM_BLOCK_SIMILARITY = 0.5     # 重复块相似度阈值
SSM_CHORUS_OVERLAP = 0.5       # 副歌重叠率阈值

# 边界检测
MIN_SONG_INTERVAL = 15.0       # 歌曲最小间隔（秒）
MIN_SONG_DURATION = 15.0       # 歌曲最小时长（秒）

# VAD 参数
VAD_THRESHOLD = 0.4            # Silero VAD 阈值
VAD_MIN_SPEECH_MS = 300        # 最小语音时长（毫秒）
VAD_MIN_SILENCE_MS = 500       # 最小静音时长（毫秒）

# Demucs 参数
DEMUCS_SEGMENT_SEC = 10        # Demucs 分段时长（秒，减少显存占用）
DEMUCS_OVERLAP = 0.25          # Demucs 重叠率

# 注意：更多常量已迁移到 config.py，此处保留向后兼容


@dataclass
class SegmentFeatures:
    """
    封装段落的所有特征和动态阈值（优化 v5.0）
    v5.0 新增：Chroma特征、谱通量、高频ZCR、Delta-MFCC、Tempogram、频谱扩展度
    """
    # ── 原始特征 ───────────────────────────────────────────────────────────
    rms: float = 0.0              # RMS 能量
    zcr: float = 0.0              # 过零率
    centroid: float = 0.0         # 频谱质心
    flatness: float = 0.0         # 频谱平坦度
    rolloff: float = 0.0          # 频谱滚降
    beat_strength: float = 0.0    # 节拍强度
    beat_regularity: float = 0.0  # 节拍规律性
    voice_ratio: float = 0.0      # 人声占比
    pitch_strength: float = 0.0   # 音高强度

    # ── v5.0 新增特征 ──────────────────────────────────────────────────────
    chroma_entropy: float = 0.0   # Chroma 熵（和声清晰度，副歌低熵=和声稳定）
    spectral_flux: float = 0.0    # 谱通量（频谱变化速率，副歌高通量）
    hf_zcr: float = 0.0           # 高频过零率（>2kHz，观众噪声特征）
    spectral_spread: float = 0.0  # 频谱扩展度（频谱宽度，观众噪声宽频带）
    delta_mfcc_energy: float = 0.0  # Delta-MFCC 能量（频谱动态变化，副歌较高）
    silence_ratio: float = 0.0    # 静默率（低能量帧比例，讲话时较高）

    # ── v6.0 新增特征 ──────────────────────────────────────────────────────
    rms_variance: float = 0.0     # RMS 帧级方差（掌声方差大，音乐方差小）
    hf_energy_ratio: float = 0.0  # 高频能量占比（>4kHz / 全频，掌声/欢呼 > 0.3）
    tempo: float = 0.0             # 估计 BPM（讲话/掌声无明显节拍）
    harmonic_ratio: float = 0.0   # 谐波比（harmonic/percussive，纯音乐高，噪声低）

    # ── 动态阈值（从 global_stats 计算）─────────────────────────────────
    # RMS 百分位
    rms_p10: float = 0.005
    rms_p15: float = 0.010
    rms_p20: float = 0.012
    rms_p25: float = 0.015
    rms_p50: float = 0.04
    rms_p60: float = 0.055
    rms_p75: float = 0.07
    rms_p90: float = 0.10
    rms_mean: float = 0.045  # 全曲 RMS 平均值

    # 过零率
    zcr_mean: float = 0.06
    zcr_p60: float = 0.08
    zcr_p75: float = 0.10

    # 频谱质心
    centroid_mean: float = 2000.0
    centroid_p50: float = 2000.0
    centroid_p60: float = 2500.0
    centroid_p75: float = 3000.0

    # 频谱平坦度
    flatness_mean: float = 0.05
    flatness_p60: float = 0.07
    flatness_p75: float = 0.09

    # 节拍
    beat_mean: float = 0.3

    # 频谱滚降
    rolloff_p75: float = 5000.0
    rolloff_p90: float = 6000.0

    # 能量标准差
    rms_std: float = 0.03
    rms_p95: float = 0.12

    # 节拍强度标准差
    beat_std: float = 0.1

    # ── 歌曲内相对能量（优化1）──────────────────────────────────────────
    song_rms_baseline: float = 0.0   # 歌曲内 RMS 均值（用于相对能量计算）
    relative_energy: float = 0.0      # 段落 RMS / 歌曲 RMS 均值（>1=高潮,<1=低潮）
    song_rms_p50: float = 0.0         # 歌曲内 RMS 中位数
    song_rms_p75: float = 0.0         # 歌曲内 RMS 75分位（副歌参考线）
    song_rms_p85: float = 0.0         # 歌曲内 RMS 85分位（强副歌参考线）

    # ── 频谱质心稳定性（优化3）──────────────────────────────────────────
    centroid_stability: float = 1.0   # 质心稳定性 (0~1, 1=极稳定)
    centroid_gradient: float = 0.0    # 质心变化梯度（Hz/s，突变点附近大）
    centroid_relative: float = 0.0     # 质心 / 歌曲均值（归一化质心）

    # ── BPM 稳态（优化3副）───────────────────────────────────────────────
    bpm_stability: float = 1.0       # BPM 稳定度 (0~1, 1=极稳定)

    # ── SSM 副歌似然（优化2）─────────────────────────────────────────────
    ssm_chorus_likelihood: float = 0.0  # SSM 返回的副歌似然 (0~1)

    # ── 结构特征（v7.1：重复性 + 位置）────────────────────────────────────
    repeat_count: int = 0              # 有多少个其他段落与此段 MFCC 相似（>=1=重复段）
    avg_similarity: float = 0.0        # 与其他段落的平均 MFCC 余弦相似度 (0~1)

    # ── 便捷属性 ───────────────────────────────────────────────────────────
    @property
    def is_high_energy(self) -> bool:
        return self.rms > self.rms_p50

    @property
    def is_very_high_energy(self) -> bool:
        return self.rms > self.rms_p75

    @property
    def has_voice(self) -> bool:
        return 0.25 < self.voice_ratio < 0.90

    @property
    def is_speech_like(self) -> bool:
        return self.voice_ratio < 0.25

    @property
    def has_stable_beat(self) -> bool:
        return self.beat_regularity > 0.3

    @property
    def zcr_above_mean(self) -> bool:
        return self.zcr > self.zcr_mean

    @property
    def flatness_above_mean(self) -> bool:
        return self.flatness > self.flatness_mean

    @property
    def is_song_high_energy(self) -> bool:
        """歌曲内相对能量是否高于均值"""
        return self.song_rms_baseline > 0 and self.relative_energy > 1.0

    @property
    def is_song_very_high_energy(self) -> bool:
        """歌曲内相对能量是否高于 85 分位"""
        return self.song_rms_p85 > 0 and self.relative_energy > (self.song_rms_p85 / max(self.song_rms_baseline, 1e-8))


@dataclass
class Segment:
    start_time: float
    end_time: float
    label: str
    confidence: float
    song_index: int
    features: Optional[Dict[str, Any]] = None
    songformer_label: str = ""  # SongFormer 英文原始标签（如 "chorus"）

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SongInfo:
    song_index: int
    song_name: str
    segments: List[Segment]
    start_time: float
    end_time: float
    bpm: Optional[float] = None
    key: Optional[str] = None
    song_title: str = ""
    song_artist: str = ""
    song_confidence: float = 0.0
    boundary_confidence: float = 0.0
    track_id: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        result['segments'] = [seg.to_dict() for seg in self.segments]
        return result

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AnalysisResult:
    singer: str
    songs: List[SongInfo]
    total_duration: float
    audio_info: Dict[str, Any]
    analysis_time: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result['songs'] = [song.to_dict() for song in self.songs]
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ── VAD 工具 ──────────────────────────────────────────────────────────────────

class VoiceActivityDetector:
    """
    语音活动检测器
    优先使用 FireRedVAD（深度学习 SOTA），其次 Silero VAD，降级到 librosa 规则
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._silero_ready = _try_load_silero()
        self._firered_ready = _try_load_firered_vad()

    def get_voice_mask(self, audio_path: str, hop_sec: float = 0.5) -> Tuple[np.ndarray, float]:
        """
        返回语音活动掩码（每 hop_sec 一个布尔值）和帧率

        Returns:
            (mask: bool array, fps: float)
        """
        if self._firered_ready:
            return self._vad_firered(audio_path, hop_sec)
        elif self._silero_ready:
            return self._vad_silero(audio_path, hop_sec)
        else:
            return self._vad_librosa(audio_path, hop_sec)

    def _vad_firered(self, audio_path: str, hop_sec: float) -> Tuple[np.ndarray, float]:
        """使用 FireRedVAD 进行语音活动检测"""
        import soundfile as sf
        global _firered_vad_model, _firered_aed_model

        wav, sr = sf.read(audio_path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        logger.info(f"[AED音频] path={audio_path}, sr={sr}, duration={len(wav)/sr:.2f}s, shape={wav.shape}")

        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
            logger.info(f"[AED音频] 重采样后 sr={sr}, duration={len(wav)/sr:.2f}s")

        result = _firered_vad_model.detect(wav, sr)
        if isinstance(result, tuple):
            result = result[0]
        timestamps = result.get('timestamps', [])

        total_sec = len(wav) / 16000
        fps = 1.0 / hop_sec
        n_frames = int(total_sec * fps) + 1
        mask = np.zeros(n_frames, dtype=bool)

        for start_sec, end_sec in timestamps:
            start_f = int(start_sec * fps)
            end_f = int(end_sec * fps)
            mask[start_f:end_f+1] = True

        return mask, fps

    def get_aed_result(self, audio_path: str) -> Optional[dict]:
        """
        使用 FireRedAED 进行音频事件检测（区分 speech/singing/music）
        仅当 FireRedVAD 可用时可用
        """
        if not self._firered_ready:
            return None
        global _firered_aed_model

        try:
            result, _ = _firered_aed_model.detect(audio_path)
            return result
        except Exception as e1:
            logger.warning(f"[AED] 直接传文件路径失败: {e1}，尝试传 (wav, sr)")
            import soundfile as sf
            wav, sr = sf.read(audio_path, dtype='float32')
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                sr = 16000
            try:
                result, _ = _firered_aed_model.detect((wav, sr))
                return result
            except Exception as e2:
                logger.warning(f"[AED] 传 (wav, sr) 也失败: {e2}")
                return None

    def _vad_silero(self, audio_path: str, hop_sec: float) -> Tuple[np.ndarray, float]:
        """使用 Silero VAD"""
        global _silero_permanently_unavailable, _silero_unavailable_reason
        import torch
        get_speech_timestamps, _, read_audio, *_ = _silero_utils

        try:
            wav = read_audio(audio_path, sampling_rate=16000)
            speech_ts = get_speech_timestamps(
                wav, _silero_model,
                sampling_rate=16000,
                threshold=0.4,
                min_speech_duration_ms=300,
                min_silence_duration_ms=500,
            )

            total_samples = len(wav)
            total_sec = total_samples / 16000
            fps = 1.0 / hop_sec
            n_frames = int(total_sec * fps) + 1
            mask = np.zeros(n_frames, dtype=bool)

            for ts in speech_ts:
                start_f = int(ts['start'] / 16000 * fps)
                end_f   = int(ts['end']   / 16000 * fps)
                mask[start_f:end_f+1] = True

            return mask, fps

        except Exception as e:
            err = str(e)
            err_l = err.lower()
            # 避免输出超长 traceback 文本污染日志
            err_short = err.splitlines()[0] if err else "unknown"
            logger.warning(f"Silero VAD 推理失败: {err_short}，降级到 librosa")
            if "torchcodec" in err_l or "torchaudio" in err_l:
                _silero_permanently_unavailable = True
                _silero_unavailable_reason = err_short[:300]
                self._silero_ready = False
            return self._vad_librosa(audio_path, hop_sec)

    def _vad_librosa(self, audio_path: str, hop_sec: float) -> Tuple[np.ndarray, float]:
        """基于 librosa 的规则 VAD"""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        hop = int(hop_sec * sr)

        # 能量
        rms = librosa.feature.rms(y=y, frame_length=hop*2, hop_length=hop)[0]
        # 过零率
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=hop*2, hop_length=hop)[0]
        # 频谱质心
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

        # 人声特征：中等能量 + 中等过零率 + 频谱质心在语音范围
        voice_mask = (
            (rms > np.percentile(rms, 20)) &
            (zcr > 0.02) & (zcr < 0.20) &
            (centroid > 300) & (centroid < 4000)
        )
        fps = sr / hop
        return voice_mask, fps


# ── 掌声检测 ──────────────────────────────────────────────────────────────────

class ApplauseDetector:
    """
    掌声/欢呼声检测器
    特征：宽频噪声、高过零率、频谱平坦、无明显音高
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 1024):
        self.sr = sample_rate
        self.hop = hop_length

    def get_applause_mask(self, y: np.ndarray) -> np.ndarray:
        """返回每帧是否为掌声的布尔掩码"""
        hop = self.hop
        sr  = self.sr

        # 过零率（掌声极高）
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=hop*2, hop_length=hop)[0]

        # 频谱平坦度（掌声接近白噪声，平坦度高）
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]

        # 频谱质心（掌声宽频，质心偏高）
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

        # 能量
        rms = librosa.feature.rms(y=y, frame_length=hop*2, hop_length=hop)[0]

        # 音调强度（掌声无明显音高）
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop)
            pitch_strength = magnitudes.max(axis=0)
            pitch_strength = pitch_strength / (pitch_strength.max() + 1e-6)
        except Exception:
            pitch_strength = np.zeros(len(zcr))

        # 对齐长度
        n = min(len(zcr), len(flatness), len(centroid), len(rms), len(pitch_strength))
        zcr       = zcr[:n]
        flatness  = flatness[:n]
        centroid  = centroid[:n]
        rms       = rms[:n]
        pitch_str = pitch_strength[:n]

        # 掌声判定：高过零率 + 高平坦度 + 低音调强度 + 有能量
        applause_mask = (
            (zcr > np.percentile(zcr, 60)) &
            (flatness > np.percentile(flatness, 50)) &
            (pitch_str < np.percentile(pitch_str, 60)) &
            (rms > np.percentile(rms, 15))
        )

        # 形态学平滑（去除孤立帧）
        min_frames = max(3, int(1.0 * sr / hop))  # 至少1秒
        applause_mask = scipy.ndimage.binary_opening(applause_mask, structure=np.ones(min_frames))
        applause_mask = scipy.ndimage.binary_closing(applause_mask, structure=np.ones(min_frames * 2))

        return applause_mask

    def get_applause_regions(self, y: np.ndarray) -> List[Tuple[float, float]]:
        """返回掌声区域列表 [(start_sec, end_sec), ...]"""
        mask = self.get_applause_mask(y)
        fps  = self.sr / self.hop
        regions = []
        in_region = False
        start = 0

        for i, v in enumerate(mask):
            if v and not in_region:
                in_region = True
                start = i
            elif not v and in_region:
                in_region = False
                regions.append((start / fps, i / fps))

        if in_region:
            regions.append((start / fps, len(mask) / fps))

        return regions


# ── 歌曲边界检测 ──────────────────────────────────────────────────────────────

class SongBoundaryDetector:
    """
    演唱会歌曲边界检测器 v2.0
    策略：
      1. 掌声区域内的能量谷值
      2. 能量包络的局部最低点（即使没有掌声）
      3. Chroma 调性跳变
      4. 节拍模式变化
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 1024):
        self.sr = sample_rate
        self.hop = hop_length
        self.applause_detector = ApplauseDetector(sample_rate, hop_length)

    def _detect_chroma_key_changes(self, chroma: np.ndarray, fps: float, duration: float) -> np.ndarray:
        """
        检测 Chroma Key 调性跳变（优化4）
        
        方法：对每帧，计算当前 Chroma 向量与前后窗口的余弦相似度。
        当相似度骤降（即调性发生显著变化），说明可能是新歌曲开始。
        
        Returns:
            scores: 每个时间点的调性跳变得分 [n_frames]，越高越可能是边界
        """
        n_frames = chroma.shape[1]
        win_sec = CONFIG.CHORMA_KEY_WINDOW_SEC  # 2.0 秒窗口
        win_frames = max(1, int(win_sec * fps))
        threshold = CONFIG.CHORMA_KEY_CHANGE_THRESHOLD  # 0.40 骤降阈值

        scores = np.zeros(n_frames)

        for i in range(win_frames, n_frames - win_frames):
            # 前后窗口的 Chroma 均值向量
            c_before = chroma[:, i-win_frames:i].mean(axis=1)
            c_after = chroma[:, i:i+win_frames].mean(axis=1)

            # 余弦相似度
            norm_b = np.linalg.norm(c_before) + 1e-8
            norm_a = np.linalg.norm(c_after) + 1e-8
            sim = np.dot(c_before, c_after) / (norm_b * norm_a)

            # 相似度骤降检测：前后差值
            # 找前后窗口内各自的局部相似度
            sims_before = []
            for j in range(i - win_frames, i - max(1, win_frames // 2)):
                cb = chroma[:, max(0, j-win_frames//2):j].mean(axis=1)
                ca = chroma[:, max(0, j-win_frames//4):j+win_frames//4].mean(axis=1)
                nb = np.linalg.norm(cb) + 1e-8
                na = np.linalg.norm(ca) + 1e-8
                sims_before.append(np.dot(cb, ca) / (nb * na))
            
            sim_mean_before = np.mean(sims_before) if sims_before else 0.8

            # 相似度骤降 = 当前相似度显著低于历史均值
            drop = sim_mean_before - sim
            if drop > threshold:
                scores[i] = drop

        # 平滑
        scores = scipy.ndimage.uniform_filter1d(scores, size=int(fps * 2))
        scores = scores / (scores.max() + 1e-6)
        return scores

    def detect_boundaries(
        self,
        y: np.ndarray,
        total_duration: float,
        progress_callback=None
    ) -> List[float]:
        """
        检测歌曲边界时间点列表 v3.0

        增强策略：
        1. 掌声区域内的能量谷值
        2. 全局能量局部最低点
        3. Chroma 调性跳变
        4. 节拍强度变化
        5. 频谱质心突变
        6. MFCC 音色特征变化（人声特征）
        7. 观众情绪变化（掌声突然增大/减小）

        Returns:
            边界时间点列表，不含 0 和 total_duration
        """
        fps = self.sr / self.hop
        win_sec = 3  # 分析窗口大小（秒）

        if progress_callback:
            progress_callback(0.05, 1.0, "计算能量包络...")

        # 1. 能量包络（每帧 RMS）
        rms = librosa.feature.rms(y=y, frame_length=self.hop*2, hop_length=self.hop)[0]
        rms_smooth = scipy.ndimage.uniform_filter1d(rms, size=int(fps * 3))  # 3秒平滑

        if progress_callback:
            progress_callback(0.10, 1.0, "检测掌声区域...")

        # 2. 掌声区域
        applause_regions = self.applause_detector.get_applause_regions(y)
        logger.info(f"检测到 {len(applause_regions)} 个掌声区域")

        # 创建掌声强度曲线（用于检测观众情绪变化）
        applause_intensity = np.zeros(len(rms))
        for ap_start, ap_end in applause_regions:
            f_start = max(0, int(ap_start * fps))
            f_end = min(len(applause_intensity), int(ap_end * fps))
            applause_intensity[f_start:f_end] = 1.0

        if progress_callback:
            progress_callback(0.15, 1.0, "计算频谱质心...")

        # 3. 频谱质心
        try:
            centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop)[0]
            # 计算质心变化率（差分 + 平滑）
            centroid_diff = np.abs(np.diff(centroid))
            centroid_diff = np.concatenate([[0], centroid_diff])
            centroid_diff = scipy.ndimage.uniform_filter1d(centroid_diff, size=int(fps * win_sec))
            centroid_diff = centroid_diff / (centroid_diff.max() + 1e-6)
        except Exception:
            centroid_diff = np.zeros(len(rms))

        if progress_callback:
            progress_callback(0.25, 1.0, "计算MFCC特征...")

        # 4. MFCC 音色特征变化（捕捉人声特征变化）
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, hop_length=self.hop)
            # 计算 MFCC 变化（相邻窗口的差异）
            mfcc_diff = np.zeros(mfcc.shape[1])
            win_frames = int(fps * win_sec)
            for i in range(win_frames, mfcc.shape[1] - win_frames):
                m1 = mfcc[:, i-win_frames:i].mean(axis=1)
                m2 = mfcc[:, i:i+win_frames].mean(axis=1)
                mfcc_diff[i] = np.linalg.norm(m1 - m2)
            mfcc_diff = scipy.ndimage.uniform_filter1d(mfcc_diff, size=int(fps * win_sec))
            mfcc_diff = mfcc_diff / (mfcc_diff.max() + 1e-6)
        except Exception:
            mfcc_diff = np.zeros(len(rms))

        if progress_callback:
            progress_callback(0.35, 1.0, "计算节拍强度...")

        # 5. 节拍强度
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop)
            beat_strength_diff = np.abs(np.diff(onset_env))
            beat_strength_diff = np.concatenate([[0], beat_strength_diff])
            beat_strength_diff = scipy.ndimage.uniform_filter1d(beat_strength_diff, size=int(fps * 2))
            beat_strength_diff = beat_strength_diff / (beat_strength_diff.max() + 1e-6)
        except Exception:
            beat_strength_diff = np.zeros(len(rms))

        if progress_callback:
            progress_callback(0.45, 1.0, "计算Chroma跳变...")

        # 6. Chroma 跳变（调性变化）
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop)
            win_frames = int(fps * 2)
            chroma_diff = np.zeros(chroma.shape[1])
            for i in range(win_frames, chroma.shape[1] - win_frames):
                c1 = chroma[:, i-win_frames:i].mean(axis=1)
                c2 = chroma[:, i:i+win_frames].mean(axis=1)
                chroma_diff[i] = np.linalg.norm(c1 - c2)
            chroma_diff = scipy.ndimage.uniform_filter1d(chroma_diff, size=int(fps * win_sec))
            chroma_diff = chroma_diff / (chroma_diff.max() + 1e-6)
        except Exception:
            chroma_diff = np.zeros(len(rms))

        if progress_callback:
            progress_callback(0.55, 1.0, "计算观众情绪变化...")

        # 7. 观众情绪变化（掌声的突变）
        try:
            # 计算掌声强度的变化率
            applause_change = np.abs(np.diff(applause_intensity))
            applause_change = np.concatenate([[0], applause_change])
            applause_change = scipy.ndimage.uniform_filter1d(applause_change, size=int(fps * win_sec))
            applause_change = applause_change / (applause_change.max() + 1e-6)
        except Exception:
            applause_change = np.zeros(len(rms))

        # 8. 计算综合边界得分
        if progress_callback:
            progress_callback(0.65, 1.0, "计算综合边界得分...")

        n_frames = len(rms)
        
        # ══════════════════════════════════════════════════════════════
        # v7.0 优化4: Chroma Key 调性跳变检测（余弦相似度骤降）
        # ══════════════════════════════════════════════════════════════
        try:
            chroma_key_scores = self._detect_chroma_key_changes(chroma, fps, y.shape[0] / self.sr)
            logger.info(f"Chroma Key 检测到 {sum(1 for s in chroma_key_scores if s > 0)} 个调性跳变点")
        except Exception as e:
            logger.debug(f"Chroma Key 检测失败: {e}")
            chroma_key_scores = np.zeros(n_frames)
        scores = np.zeros(n_frames)

        # 归一化 rms（找谷值，用反向）
        rms_norm = 1 - (rms_smooth - rms_smooth.min()) / (rms_smooth.max() - rms_smooth.min() + 1e-6)

        # 信号1: 掌声区域内的低能量点（权重高）
        for ap_start, ap_end in applause_regions:
            f_start = max(0, int(ap_start * fps))
            f_end = min(n_frames, int(ap_end * fps))
            if f_end > f_start:
                local_rms = rms_norm[f_start:f_end]
                if local_rms.max() > local_rms.min():
                    local_rms = 1 - (local_rms - local_rms.min()) / (local_rms.max() - local_rms.min() + 1e-6)
                scores[f_start:f_end] += local_rms * 3.0

        # 信号2: Chroma 跳变
        if len(chroma_diff) <= n_frames:
            scores[:len(chroma_diff)] += chroma_diff * 2.0

        # 信号3: MFCC 音色变化（新增）
        if len(mfcc_diff) <= n_frames:
            scores[:len(mfcc_diff)] += mfcc_diff * 2.5

        # 信号4: 频谱质心突变（新增）
        if len(centroid_diff) <= n_frames:
            scores[:len(centroid_diff)] += centroid_diff * 1.8

        # 信号5: 节拍强度变化
        if len(beat_strength_diff) <= n_frames:
            scores[:len(beat_strength_diff)] += beat_strength_diff * 1.2

        # 信号6: 观众情绪变化（新增）
        if len(applause_change) <= n_frames:
            scores[:len(applause_change)] += applause_change * 2.0

        # 信号7: Chroma Key 调性跳变（余弦相似度骤降，优化4）
        if len(chroma_key_scores) <= n_frames:
            scores[:len(chroma_key_scores)] += chroma_key_scores * 3.0

        # 信号8: 全局能量谷值
        try:
            valleys = signal.argrelmin(rms_smooth, order=int(fps * 15))[0]
            for v in valleys:
                if v < n_frames:
                    scores[v] += 0.8
        except Exception:
            pass

        if progress_callback:
            progress_callback(0.80, 1.0, "寻找峰值...")

        # 6. 找峰值（v7.0: AND 融合策略）
        # 策略：每种信号单独计算局部最大值，然后取它们的交集（AND）
        #        交集外的信号单独高分点作为 OR 补充（保证召回）
        min_song_frames = int(30 * fps)  # 最小30秒
        boundary_candidates = []

        # 6a. 综合得分局部最大值
        try:
            _order = min(min_song_frames, len(scores) - 1)
            peaks = signal.argrelmax(scores, order=_order)[0] if len(scores) > _order else np.array([])
        except (IndexError, ValueError):
            peaks = np.array([])

        # 6b. 各信号局部最大值（AND 融合）
        try:
            rms_peaks = signal.argrelmax(rms_norm, order=int(fps * 5))[0]
        except:
            rms_peaks = np.array([])

        try:
            chroma_peaks = signal.argrelmax(chroma_diff, order=min_song_frames)[0]
        except:
            chroma_peaks = np.array([])

        try:
            mfcc_peaks = signal.argrelmax(mfcc_diff, order=min_song_frames)[0]
        except:
            mfcc_peaks = np.array([])

        try:
            key_peaks = signal.argrelmax(chroma_key_scores, order=int(fps * 5))[0]
        except:
            key_peaks = np.array([])

        # 6c. AND 融合：同时被多种信号认可的点为高置信边界
        def frames_to_times(peaks_arr, fps_arr, min_t=30, max_t_offset=30):
            result = []
            for p in peaks_arr:
                t = p / fps_arr
                if min_t <= t <= total_duration - max_t_offset:
                    result.append((round(t * fps_arr), t))
            return result

        rms_times = frames_to_times(rms_peaks, fps)
        chroma_times = frames_to_times(chroma_peaks, fps)
        mfcc_times = frames_to_times(mfcc_peaks, fps)
        key_times = frames_to_times(key_peaks, fps)

        # 高置信边界：至少 2 种信号在 3 秒内同时指向
        tolerance_frames = int(3 * fps)

        def merge_nearby(times_list, tol_f):
            """将 tolerance 内的时间点合并"""
            if not times_list:
                return []
            merged = []
            current_group = [times_list[0]]
            for item in times_list[1:]:
                if item[0] - current_group[-1][0] <= tol_f:
                    current_group.append(item)
                else:
                    merged.append(current_group)
                    current_group = [item]
            merged.append(current_group)
            return merged

        for group in merge_nearby(rms_times + chroma_times + mfcc_times + key_times, tolerance_frames):
            times_in_group = [t for (_, t) in group]
            signal_count = 0
            for t in times_in_group:
                t_frame = int(t * fps)
                if any(abs(int(rt * fps) - t_frame) <= tolerance_frames for rt in [r for (_, r) in rms_times]):
                    signal_count += 1
                if any(abs(int(ct * fps) - t_frame) <= tolerance_frames for ct in [c for (_, c) in chroma_times]):
                    signal_count += 1
                if any(abs(int(mt * fps) - t_frame) <= tolerance_frames for mt in [m for (_, m) in mfcc_times]):
                    signal_count += 1
                if any(abs(int(kt * fps) - t_frame) <= tolerance_frames for kt in [k for (_, k) in key_times]):
                    signal_count += 1
            if signal_count >= 2:  # AND: 至少2种信号认可
                best_t = group[0][1]
                # 查找综合得分
                peak_frame = int(best_t * fps)
                best_score = scores[peak_frame] if peak_frame < len(scores) else 0
                boundary_candidates.append({
                    'time': best_t,
                    'score': best_score,
                    'confidence': 'high',
                })

        # 6d. 补充召回：综合得分极高但不在 AND 组里的点（OR 兜底）
        for peak in peaks:
            t = peak / fps
            if t < 30 or total_duration - t < 30:
                continue
            # 检查是否已在 AND 组
            already_in = any(abs(c['time'] - t) < 5 for c in boundary_candidates)
            if not already_in and scores[peak] > np.percentile(scores[scores > 0], 85):
                boundary_candidates.append({
                    'time': t,
                    'score': scores[peak],
                    'confidence': 'medium',
                })

        # 按得分排序
        boundary_candidates.sort(key=lambda x: x['score'], reverse=True)

        # 7. 贪婪选取：确保间隔至少15秒（降低阈值以检测更多边界）
        boundaries = []
        last_time = 0
        min_interval = 15.0  # 从25秒降低到15秒

        for cand in boundary_candidates:
            if cand['time'] - last_time >= min_interval:
                if total_duration - cand['time'] >= 15:  # 结尾保留15秒
                    boundaries.append(cand['time'])
                    last_time = cand['time']
                    conf = cand.get('confidence', 'n/a')
                    logger.info(f"歌曲边界: {cand['time']:.1f}s (得分: {cand['score']:.2f}, 置信: {conf})")

        # 8. 如果边界太少（<5个），用能量谷值兜底
        expected_songs = max(5, int(total_duration / 180))  # 预估歌曲数，每首约3分钟
        if len(boundaries) < expected_songs and total_duration > 180:
            logger.warning(f"边界数量太少 ({len(boundaries)})，使用能量谷值兜底")
            fallback = self._fallback_energy_boundaries(rms_smooth, fps, total_duration, min_song_sec=15)  # 降低到15秒
            for t in fallback:
                if t not in boundaries and (len(boundaries) == 0 or abs(t - boundaries[-1]) > 15):
                    boundaries.append(t)
            boundaries.sort()

        if progress_callback:
            progress_callback(1.0, 1.0, f"检测到 {len(boundaries)} 个歌曲边界")

        logger.info(f"最终边界: {[f'{b:.1f}s' for b in boundaries]}")
        return boundaries

    def _fallback_energy_boundaries(
        self,
        rms_smooth: np.ndarray,
        fps: float,
        total_duration: float,
        min_song_sec: float = 15.0
    ) -> List[float]:
        """兜底：在能量局部最低点切割 - 增强版"""
        min_frames = int(min_song_sec * fps)
        # 寻找局部最小值，邻域10秒（更敏感）
        try:
            _order = min(int(fps * 10), len(rms_smooth) - 1)
            valleys = signal.argrelmin(rms_smooth, order=_order)[0] if len(rms_smooth) > _order else np.array([])
        except (IndexError, ValueError):
            valleys = np.array([])
        boundaries = []
        last = 0

        for v in valleys:
            t = v / fps
            if v - last * fps < min_frames:
                continue
            if total_duration - t < min_song_sec:
                continue
            boundaries.append(t)
            last = v

        return boundaries


# ── SSM 结构分析器 (Self-Similarity Matrix) ────────────────────────────────

class StructureAnalyzer:
    """
    基于自相似矩阵 (SSM) 的音乐结构分析器 v4.0
    
    核心思想：
    - 副歌在歌曲中会出现多次，且每次的旋律/和声特征相似
    - 通过计算音频帧之间的相似度，构建 SSM
    - SSM 中的对角线平行块 → 重复段落（副歌）
    
    特征：
    - MFCC：捕捉音色/旋律特征
    - Chroma：捕捉和声/调性特征
    - 将两者融合，提高重复检测准确性
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sr = sample_rate
        self.hop = hop_length
        self.fps = sample_rate / hop_length

    def extract_features(self, y: np.ndarray) -> np.ndarray:
        """
        提取用于 SSM 的特征向量（MFCC + Chroma 融合）
        
        Args:
            y: 音频数据
            
        Returns:
            features: [n_features, n_frames] 的特征矩阵
        """
        # MFCC（捕捉音色变化）
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=20, hop_length=self.hop)
        # 取前 13 个 MFCC（主要的音色特征）
        mfcc = mfcc[:13]
        
        # Chroma（捕捉和声/调性）
        chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop)
        
        # Delta（变化率）- 捕捉旋律动态
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 融合特征：MFCC + Chroma + Delta
        features = np.vstack([mfcc, chroma, mfcc_delta])
        
        # L2 归一化（避免幅度差异影响相似度）
        norms = np.linalg.norm(features, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms
        
        return features

    def compute_ssm(self, features: np.ndarray) -> np.ndarray:
        """
        计算自相似矩阵
        
        Args:
            features: [n_features, n_frames] 特征矩阵
            
        Returns:
            ssm: [n_frames, n_frames] 自相似矩阵，值越大越相似
        """
        n_frames = features.shape[1]
        
        # 余弦相似度矩阵
        # SSM[i,j] = dot(features[:,i], features[:,j])
        ssm = np.dot(features.T, features)
        
        # 转换为 0-1 范围（0=完全不相似，1=完全相似）
        np.clip(ssm, 0, 1, out=ssm)
        
        return ssm

    def binarize_ssm(self, ssm: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        二值化 SSM，突出显示相似区域
        
        Args:
            ssm: 原始相似度矩阵
            threshold: 二值化阈值（相对值，使用百分位）
            
        Returns:
            binary_ssm: 二值化矩阵
        """
        # 使用动态阈值（矩阵值的 70% 百分位）
        dynamic_threshold = np.percentile(ssm[ssm > 0], SSM_PERCENTILE_THRESHOLD)
        binary = (ssm > dynamic_threshold).astype(float)
        
        # 形态学处理：去除孤立点
        binary = scipy.ndimage.binary_opening(binary, structure=np.ones((3, 3)))
        binary = scipy.ndimage.binary_closing(binary, structure=np.ones((3, 3)))
        
        return binary

    def find_repeating_blocks(self, ssm: np.ndarray, min_duration: float = SSM_MIN_BLOCK_DURATION) -> List[Dict]:
        """
        从 SSM 中找出重复的块（副歌候选区域）v4.0（优化版）
        
        原理：
        - 副歌在歌曲中出现多次
        - 在 SSM 中，每个重复出现的位置会形成平行于对角线的方块
        
        优化：减少计算复杂度，使用向量化操作
        """
        n = ssm.shape[0]
        min_frames = int(min_duration * self.fps)
        
        # v4.0 优化：简化版对角线累加
        # 只检查一定范围内的对角线偏移（节省计算）
        diagonal_accumulator = np.zeros(n)
        max_offset = min(n // 2, min_frames * 2)
        
        for offset in range(-max_offset, max_offset + 1):
            diag = np.diag(ssm, k=offset)
            # 归一化并累加
            if diag.size > 0 and diag.max() > 0 and np.isfinite(diag.max()):
                diag_norm = diag / diag.max()
                # 根据offset的正负，确定累加的位置
                if offset >= 0:
                    # offset >= 0: diag对应ssm[i, i+offset]，累加到位置[offset:]
                    start_idx = offset
                    end_idx = start_idx + len(diag_norm)
                    if end_idx <= n:
                        diagonal_accumulator[start_idx:end_idx] += diag_norm
                else:
                    # offset < 0: diag对应ssm[i-offset, i]，累加到位置[:n+offset]
                    end_idx = n + offset  # offset是负数，所以是减
                    start_idx = end_idx - len(diag_norm)
                    if start_idx >= 0:
                        diagonal_accumulator[start_idx:end_idx] += diag_norm
            else:
                logger.debug(f"跳过无效的对角线: offset={offset}, diag.size={diag.size}, diag.max()={diag.max() if diag.size > 0 else 'N/A'}")
        
        # 归一化
        if diagonal_accumulator.max() > 0:
            diagonal_accumulator /= diagonal_accumulator.max()
        
        # 找峰值（高重复区域）
        peaks, _ = signal.find_peaks(
            diagonal_accumulator,
            distance=min_frames,
            height=0.3
        )
        
        repeat_regions = []
        for peak in peaks:
            # 扩展为区间
            start = max(0, peak - min_frames // 2)
            end = min(n, peak + min_frames // 2)
            
            # 计算该区域的平均相似度
            region_ssm = ssm[start:end, start:end]
            similarity = region_ssm.mean() if region_ssm.size > 0 else 0
            
            # 估计出现次数
            occurrences = self._count_occurrences(ssm, start, end)
            
            repeat_regions.append({
                'frame_start': start,
                'frame_end': end,
                'start_time': start / self.fps,
                'end_time': end / self.fps,
                'similarity': similarity,
                'occurrences': occurrences,
                'peak_score': diagonal_accumulator[peak],
            })
        
        # 按相似度排序
        repeat_regions.sort(key=lambda x: x['similarity'], reverse=True)
        
        return repeat_regions

    def _count_occurrences(self, ssm: np.ndarray, block_start: int, block_end: int) -> int:
        """
        计算某个块在整首歌中出现的次数 v5.0（优化版）
        
        优化点：
        - 添加参数验证
        - 使用向量化操作，减少np.diag调用
        - 增大步长，减少迭代次数
        """
        # 1. 参数验证
        n = ssm.shape[0]
        
        # 验证block范围
        if block_start >= block_end:
            logger.warning(f"无效的块范围: block_start={block_start}, block_end={block_end}")
            return 1  # 至少自身出现一次
        
        if block_start < 0 or block_end > n:
            logger.error(f"块范围超出边界: [{block_start}, {block_end}], SSM shape=({n}, {n})")
            return 1
        
        block_size = block_end - block_start
        
        # 块大小验证
        if block_size <= 0 or block_size > n:
            logger.warning(f"无效的块大小: {block_size}, SSM shape=({n}, {n})")
            return 1
        
        occurrences = 1  # 至少出现一次（自身）
        
        # 2. 优化：预计算行索引（避免在循环中重复创建）
        row_indices = np.arange(block_start, block_end)
        
        # 3. 优化：增大步长，减少迭代次数（从block_size//2改为block_size）
        max_offset = min(n - block_end + 1, n // 2)  # 确保不超出边界
        step_size = max(1, block_size // 2)  # 至少为1
        
        # 4. 检测是否有其他对角线上的相似区域
        for offset in range(block_size, max_offset, step_size):
            try:
                # 优化：直接数组索引，避免np.diag创建中间数组
                col_indices = row_indices + offset
                
                # 边界检查（提前退出条件）
                if col_indices[-1] >= n:
                    logger.debug(f"达到边界，停止检查: offset={offset}, max_offset={max_offset}")
                    break
                
                # 直接向量化索引（比np.diag更快）
                diag_values = ssm[row_indices, col_indices]
                
                # 验证数据有效性
                if diag_values.size == 0:
                    logger.debug(f"空的对角线数据: offset={offset}")
                    continue
                
                # 计算均值并检查阈值
                diag_mean = diag_values.mean()
                if diag_mean > SSM_SIMILARITY_THRESHOLD:  # 相似度阈值判断
                    occurrences += 1
                    logger.debug(f"找到重复块: offset={offset}, similarity={diag_mean:.3f}")
                    
            except IndexError as e:
                # 数组索引越界，这是预期的边界情况
                logger.debug(f"对角线索引越界: offset={offset}, n={n}, error={e}")
                break  # 达到边界，提前退出
            except Exception as e:
                # 其他未预期的错误，记录但继续处理
                logger.warning(f"检测对角线相似性时出错: offset={offset}, error={e}")
        
        logger.debug(f"块[{block_start}:{block_end}]出现次数: {occurrences}")
        return occurrences

    def detect_chorus_by_energy_envelope(
        self,
        y: np.ndarray,
        sr: int = 22050,
        hop_length: int = 512,
        min_duration: float = 20.0,
    ) -> List[Dict]:
        """
        简化版副歌检测（当 SSM 不可用时）
        
        原理：副歌通常是歌曲中能量最高的段落
        1. 计算能量包络
        2. 找局部峰值（能量高于平均值的区域）
        3. 这些区域标记为副歌候选
        """
        fps = sr / hop_length
        
        # 能量包络
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_smooth = scipy.ndimage.uniform_filter1d(rms, size=int(fps * 2))  # 减少平滑窗口，增加敏感度
        
        # 全局阈值
        rms_mean = rms_smooth.mean()
        rms_p50 = np.percentile(rms_smooth, 50)  # 降低阈值到 p50
        
        # 找局部峰值（能量 > p50 的区域）
        peaks, properties = signal.find_peaks(
            rms_smooth,
            height=rms_p50,
            distance=int(fps * 15),  # 减少间隔到 15 秒
            prominence=0.03,  # 降低突出度要求
        )
        
        chorus_regions = []
        peak_heights = properties.get('peak_heights', np.array([0.5] * len(peaks)))
        
        for i, peak in enumerate(peaks):
            # 扩展为区间（前后各 12-18 秒）
            start_frame = max(0, peak - int(fps * 12))
            end_frame = min(len(rms_smooth), peak + int(fps * 18))
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            if end_time - start_time >= min_duration:
                confidence = float(peak_heights[i] / rms_smooth.max()) if len(peak_heights) > i else 0.5
                chorus_regions.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': confidence,
                    'occurrences': 1,
                })
        
        logger.info(f"能量包络副歌检测: {len(chorus_regions)} 个候选区域")
        return chorus_regions

    def analyze(self, y: np.ndarray, progress_callback=None) -> Dict[str, Any]:
        """
        完整的 SSM 结构分析
        
        Returns:
            结构分析结果，包含：
            - features: 特征矩阵
            - ssm: 原始相似度矩阵
            - repeating_blocks: 重复块（副歌候选）
            - chorus_regions: 识别的副歌区域
            - non_repeating_regions: 非重复区域（主歌/前奏/间奏候选）
        """
        if progress_callback:
            progress_callback(0.0, 1.0, "提取音频特征...")
        
        # 1. 下采样（如果音频太长，节省内存）
        # 目标：帧数不超过 20000
        n_seconds = len(y) / self.fps
        MAX_SSM_DURATION = 900  # 放宽到 15 分钟
        MAX_FRAMES = 20000
        
        # 计算需要的下采样率
        if n_seconds > MAX_SSM_DURATION:
            downsample_rate = int(n_seconds / MAX_SSM_DURATION) + 1
        else:
            downsample_rate = 1
        
        if downsample_rate > 1:
            logger.info(f"SSM 下采样: {downsample_rate}x（原始 {n_seconds:.0f}秒）")
            y_down = y[::downsample_rate]
        else:
            y_down = y
        
        # 特征提取
        features = self.extract_features(y_down)
        n_frames = features.shape[1]
        total_duration = len(y_down) / self.fps
        
        if progress_callback:
            progress_callback(0.3, 1.0, "计算自相似矩阵...")
        
        # 2. 计算 SSM
        ssm = self.compute_ssm(features)
        
        if progress_callback:
            progress_callback(0.5, 1.0, "检测重复结构...")
        
        # 3. 二值化
        binary_ssm = self.binarize_ssm(ssm)
        
        # 4. 找重复块
        repeat_blocks = self.find_repeating_blocks(binary_ssm, min_duration=SSM_MIN_CHORUS_DURATION)  # 降低最小时长要求
        
        if progress_callback:
            progress_callback(0.8, 1.0, "分析重复模式...")
        
        # 5. 识别副歌区域
        chorus_regions = []
        for block in repeat_blocks:
            if block['occurrences'] >= 1.2 and block['similarity'] > 0.2:  # 进一步降低阈值要求
                chorus_regions.append({
                    'start_time': block['start_time'],
                    'end_time': block['end_time'],
                    'confidence': block['similarity'],
                    'occurrences': block['occurrences'],
                })
        
        # 如果没有识别到副歌区域，使用能量包络检测作为兜底
        if not chorus_regions:
            try:
                chorus_regions = self.detect_chorus_by_energy_envelope(y, self.sr, min_duration=10.0)
                logger.info(f"SSM 未检测到副歌，使用能量包络兜底: {len(chorus_regions)} 个区域")
            except Exception as e:
                logger.warning(f"能量包络检测失败: {e}")
        
        # 6. 识别非重复区域
        non_repeat_regions = []
        used_frames = set()
        
        # 按时间顺序处理
        sorted_blocks = sorted(repeat_blocks, key=lambda x: x['start_time'])
        
        prev_end = 0
        for block in sorted_blocks:
            block_start_f = block['frame_start']
            block_end_f = block['frame_end']
            
            # 非重复区域
            if block_start_f > prev_end + int(5 * self.fps):  # 至少间隔 5 秒
                non_repeat_regions.append({
                    'start_time': prev_end / self.fps,
                    'end_time': block_start_f / self.fps,
                    'is_chorus': False,
                })
            
            used_frames.update(range(block_start_f, block_end_f))
            prev_end = block_end_f
        
        # 尾部非重复区域
        if prev_end < n_frames - int(10 * self.fps):
            non_repeat_regions.append({
                'start_time': prev_end / self.fps,
                'end_time': total_duration,
                'is_chorus': False,
            })
        
        if progress_callback:
            progress_callback(1.0, 1.0, f"SSM 分析完成：发现 {len(chorus_regions)} 个重复区域")
        
        logger.info(f"SSM 分析: {len(chorus_regions)} 个副歌区域, {len(non_repeat_regions)} 个非重复区域")
        
        return {
            'features': features,
            'ssm': ssm,
            'binary_ssm': binary_ssm,
            'repeating_blocks': repeat_blocks,
            'chorus_regions': chorus_regions,
            'non_repeating_regions': non_repeat_regions,
            'n_frames': n_frames,
            'fps': self.fps,
        }

    def merge_with_classification(
        self,
        ssm_result: Dict,
        segments: List[Segment],
    ) -> List[Segment]:
        """
        将 SSM 分析结果与基于特征的分类结果融合
        
        策略（v6.1 重写）：
        - 特征分类是主要依据（尊重 classify 的判断）
        - SSM 副歌区域作为**辅助增强**，仅在特征分类为音乐类标签时升级为 chorus
        - 如果 SSM 副歌区域过多（>30%覆盖率），视为兜底失效，不使用 SSM 覆盖
        - speech/talk/crowd/audience/silence 等 Stage 1 标签永远不被 SSM 覆盖
        """
        if not ssm_result or not ssm_result.get('chorus_regions'):
            return segments
        
        chorus_regions = ssm_result['chorus_regions']
        fps = ssm_result.get('fps', 1.0)
        total_frames = ssm_result.get('n_frames', 0)
        total_duration = total_frames / fps if fps > 0 and total_frames > 0 else 0
        
        # 计算 chorus_regions 覆盖率：如果覆盖超过 60% 时间，视为兜底失效
        if total_duration > 0:
            covered_frames = set()
            for cr in chorus_regions:
                s = int(cr['start_time'] * fps)
                e = int(cr['end_time'] * fps)
                covered_frames.update(range(s, min(e, total_frames)))
            coverage_ratio = len(covered_frames) / total_frames if total_frames > 0 else 0
            
            if coverage_ratio > 0.6:
                logger.warning(
                    f"SSM chorus_regions 覆盖 {coverage_ratio*100:.0f}% (>60%)，"
                    f"视为兜底失效，跳过 SSM 融合"
                )
                return segments
        
        # Stage 1 标签：这些标签不应被 SSM 覆盖
        STAGE1_LABELS = {LABEL_SILENCE, LABEL_SPEECH, LABEL_TALK, LABEL_CROWD, LABEL_AUDIENCE}
        
        updated_segments = []
        
        for seg in segments:
            seg_start = seg.start_time
            seg_end = seg.end_time
            original_label = seg.label
            
            # 检查是否与副歌区域重叠
            is_chorus_region = False
            chorus_confidence = 0.0
            
            for chorus in chorus_regions:
                overlap_start = max(seg_start, chorus['start_time'])
                overlap_end = min(seg_end, chorus['end_time'])
                
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / seg.duration
                    if overlap_ratio > 0.5:
                        is_chorus_region = True
                        chorus_confidence = max(chorus_confidence, chorus['confidence'] * overlap_ratio)
            
            # 决策逻辑
            if (is_chorus_region 
                and chorus_confidence > 0.3
                and original_label not in STAGE1_LABELS):
                # 仅当特征分类为音乐类标签（verse/intro/interlude/solo/chorus/outro/other）
                # 且与 SSM 副歌区域高度重叠时，才升级为 chorus
                if original_label in (LABEL_VERSE, LABEL_INTERLUDE, LABEL_SOLO, LABEL_OTHER):
                    # 从 verse/interlude/solo 升级为 chorus（SSM 认为这里在重复）
                    updated_segments.append(Segment(
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        label=LABEL_CHORUS,
                        confidence=max(seg.confidence, chorus_confidence),
                        song_index=seg.song_index,
                        features={'ssm_chorus': True, 'ssm_confidence': chorus_confidence},
                    ))
                    continue
                elif original_label == LABEL_CHORUS:
                    # 已经是 chorus，增强置信度
                    updated_segments.append(Segment(
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        label=LABEL_CHORUS,
                        confidence=max(seg.confidence, chorus_confidence),
                        song_index=seg.song_index,
                        features={'ssm_chorus': True, 'ssm_confidence': chorus_confidence},
                    ))
                    continue
                # intro/outro 保留原有标签（位置优先）
            
            # 其他情况：保留原有分类，根据位置微调
            updated_seg = self._adjust_label_by_position(seg, ssm_result)
            updated_segments.append(updated_seg)
        
        return updated_segments

    def _adjust_label_by_position(
        self,
        segment: Segment,
        ssm_result: Dict,
    ) -> Segment:
        """
        根据位置调整标签：
        - 开头 → 可能是 intro
        - 中间 → 可能是 verse
        - 结尾 → 可能是 outro
        """
        total_duration = ssm_result['n_frames'] / ssm_result['fps']
        
        # 不在 SSM 副歌区域内的片段，根据位置调整
        relative_position = segment.start_time / total_duration
        
        adjusted_label = segment.label
        
        # 如果原本被分类为主歌，但位置靠前，可能是 intro
        if segment.label == LABEL_VERSE:
            if relative_position < 0.15 and segment.duration < 30:
                # 开头 15%，且时长较短 → 可能是前奏
                adjusted_label = LABEL_INTRO
            elif relative_position > 0.85:
                # 结尾 15% → 可能是尾奏
                adjusted_label = LABEL_OUTRO
        
        if adjusted_label != segment.label:
            logger.info(f"SSM 位置调整: {segment.start_time:.1f}s-{segment.end_time:.1f}s "
                       f"{segment.label} → {adjusted_label}")
        
        return Segment(
            start_time=segment.start_time,
            end_time=segment.end_time,
            label=adjusted_label,
            confidence=segment.confidence,
            song_index=segment.song_index,
            features=segment.features or {},
        )


# ── 段落分类器 ────────────────────────────────────────────────────────────────

class SegmentClassifier:
    """
    5类段落分类器 v3.0
    类别：前奏 / 主歌 / 副歌 / 观众合唱 / 歌手讲话

    D. 动态归一化：
        - 通过 set_global_stats() 注入全局统计量（rms_p10/p50/p90、zcr_mean 等）
        - 分类阈值改为相对全局百分位的比较，而非写死的绝对值
        - 不同录音质量/音量的视频都能得到稳定分类
    """

    def __init__(self, sample_rate: int = 22050, global_stats: Optional[Dict] = None):
        self.sr = sample_rate
        self.global_stats: Dict[str, float] = global_stats or {}

    def set_global_stats(self, stats: Dict):
        """
        设置全局统计量（AudioAnalyzer 分析完整音频后调用）。
        会与已有的 global_stats 合并，只覆盖传入的字段，保留其余字段。

        期望的 key：
            rms_p10, rms_p15, rms_p25, rms_p50, rms_p60, rms_p75, rms_p90, rms_p95  - RMS 百分位
            rms_mean, rms_std                                                       - RMS 统计
            zcr_mean, zcr_p60, zcr_p75                                              - 过零率统计
            centroid_mean, centroid_p50, centroid_p60, centroid_p75                - 频谱质心统计
            flatness_mean, flatness_p60, flatness_p75                              - 频谱平坦度统计
            rolloff_p75, rolloff_p90                                                - 频谱滚降统计
            beat_strength_mean, beat_strength_std                                   - 节拍强度统计
            hf_zcr_mean, hf_zcr_p60, hf_zcr_p75                                    - 高频ZCR统计
            chroma_entropy_mean, chroma_entropy_p50, chroma_entropy_p75            - Chroma熵统计
            spectral_flux_mean, spectral_flux_p75                                  - 谱通量统计
            bandwidth_mean, bandwidth_p75                                          - 频谱扩展度统计
        """
        # 合并模式：只覆盖传入的字段，保留已有的其余字段
        if not hasattr(self, 'global_stats') or self.global_stats is None:
            self.global_stats = {}
        self.global_stats.update(stats)
        logger.info(
            f"[动态归一化] rms: p10={self.global_stats.get('rms_p10',0):.4f} "
            f"p15={self.global_stats.get('rms_p15',0):.4f} "
            f"p50={self.global_stats.get('rms_p50',0):.4f} p90={self.global_stats.get('rms_p90',0):.4f}  "
            f"zcr_mean={self.global_stats.get('zcr_mean',0):.4f}  "
            f"centroid_mean={self.global_stats.get('centroid_mean',0):.0f}Hz  "
            f"flatness_mean={self.global_stats.get('flatness_mean',0):.4f}"
        )

    # ── 公开分类接口 ──────────────────────────────────────────────────────────

    def classify(
        self,
        y: np.ndarray,
        voice_mask_fps: Optional[Tuple[np.ndarray, float]] = None,
        vocals_y: Optional[np.ndarray] = None,
        context: Optional[Dict] = None,
    ) -> Tuple[str, float]:
        """
        对一段音频进行分类

        Args:
            y:              原始（混合）音频数据，用于能量/节拍特征
            voice_mask_fps: (VAD掩码, fps) 可选
            vocals_y:       Demucs 分离的人声轨（可选），用于更准确的 voice_ratio

        Returns:
            (label, confidence)
        """
        if len(y) < self.sr * 0.5:
            return LABEL_SILENCE, 1.0

        sr = self.sr
        n_fft = min(2048, len(y) // 4)
        if n_fft < 64:
            return LABEL_OTHER, 0.5
        hop = n_fft // 4

        try:
            ctx = context or {}
            # ── 基础特征（全轨）──────────────────────────────────────────────
            rms_val = float(np.sqrt(np.mean(y ** 2)))

            if rms_val < 1e-4:
                return LABEL_SILENCE, 0.95

            zcr = float(librosa.feature.zero_crossing_rate(
                y, frame_length=n_fft, hop_length=hop)[0].mean())

            centroid = float(librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0].mean())

            flatness = float(librosa.feature.spectral_flatness(
                y=y, n_fft=n_fft, hop_length=hop)[0].mean())

            rolloff = float(librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0].mean())

            # MFCC（前6个系数）
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6, n_fft=n_fft, hop_length=hop)
            mfcc_mean = mfcc.mean(axis=1)

            # 节拍强度
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
                beat_strength = float(onset_env.mean())
                beat_regularity = float(1.0 - onset_env.std() / (onset_env.mean() + 1e-6))
                beat_regularity = max(0.0, min(1.0, beat_regularity))
            except Exception:
                beat_strength = 0.0
                beat_regularity = 0.0

            # 音调强度
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
                pitch_strength = float(magnitudes.max(axis=0).mean())
                pitch_strength_norm = pitch_strength / (np.abs(y).max() + 1e-6)
            except Exception:
                pitch_strength_norm = 0.0

            # ── v5.0 新增特征 ──────────────────────────────────────────────────
            # Chroma 熵（和声清晰度）
            chroma_entropy_val = 0.0
            try:
                chroma_seg = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
                ch_entropies = []
                for i in range(chroma_seg.shape[1]):
                    ch = chroma_seg[:, i]
                    ch_norm = ch / (ch.sum() + 1e-8)
                    ch_norm = ch_norm[ch_norm > 0.01]
                    if len(ch_norm) > 0:
                        ce = -np.sum(ch_norm * np.log2(ch_norm + 1e-10))
                        ch_entropies.append(ce)
                if ch_entropies:
                    chroma_entropy_val = float(np.mean(ch_entropies))
            except Exception:
                pass

            # 谱通量（频谱变化速率）
            spectral_flux_val = 0.0
            try:
                spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
                flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
                spectral_flux_val = float(flux.mean())
            except Exception:
                pass

            # 高频ZCR（>2kHz 频段，观众噪声特征）
            hf_zcr_val = 0.0
            try:
                from scipy.signal import butter, filtfilt
                nyq = sr / 2
                b, a = butter(4, 2000 / nyq, btype='high')
                y_hf = filtfilt(b, a, y)
                hf_zcr_val = float(librosa.feature.zero_crossing_rate(
                    y_hf, frame_length=n_fft, hop_length=hop
                )[0].mean())
            except Exception:
                pass

            # 频谱扩展度（频谱宽度）
            spectral_spread_val = 0.0
            try:
                bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
                spectral_spread_val = float(bw.mean())
            except Exception:
                pass

            # Delta-MFCC 能量（频谱动态变化）
            delta_mfcc_energy_val = 0.0
            try:
                mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
                mfcc_delta = librosa.feature.delta(mfcc_full)
                delta_mfcc_energy_val = float(np.sqrt(np.mean(mfcc_delta**2)))
            except Exception:
                pass

            # ── A. 人声占比：优先用 Demucs 人声轨 ───────────────────────────
            voice_ratio = self._get_voice_ratio(y, sr, voice_mask_fps, vocals_y)

            # D. 动态阈值（基于全局统计量）- 必须在计算 silence_ratio 之前
            gs = self.global_stats

            # 静默率（低能量帧比例）
            silence_ratio_val = 0.0
            rms_frames_arr = None
            try:
                rms_frames_arr = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
                silence_ratio_val = float(np.mean(rms_frames_arr < gs.get('rms_p10', 0.005)))
            except Exception:
                pass

            # ── v6.0 新增特征 ──────────────────────────────────────────────────
            # RMS 帧级方差（掌声方差大，音乐方差小）
            rms_variance_val = 0.0
            try:
                if rms_frames_arr is not None:
                    rms_variance_val = float(np.var(rms_frames_arr))
            except Exception:
                pass

            # 高频能量占比（>4kHz，掌声/欢呼显著）
            hf_energy_ratio_val = 0.0
            try:
                spec_full = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
                n_bins = spec_full.shape[0]
                hf_bin = int(4000 / (sr / n_fft))  # 4kHz 对应的频率 bin
                if hf_bin < n_bins:
                    total_energy = float(np.mean(spec_full ** 2))
                    hf_energy = float(np.mean(spec_full[hf_bin:, :] ** 2))
                    if total_energy > 0:
                        hf_energy_ratio_val = hf_energy / total_energy
            except Exception:
                pass

            # 谐波比（harmonic / total，纯音乐高，噪声低）
            harmonic_ratio_val = 0.0
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                h_energy = float(np.mean(y_harmonic ** 2))
                p_energy = float(np.mean(y_percussive ** 2))
                total = h_energy + p_energy + 1e-8
                harmonic_ratio_val = h_energy / total
            except Exception:
                pass

            # BPM 估计
            tempo_val = 0.0
            try:
                if len(y) > sr * 3:  # 至少3秒才能估计
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
                    tempo_val = float(tempo)
            except Exception:
                pass

            # ── v4.0 优化：使用 SegmentFeatures 封装所有参数 ──────────────────
            features = SegmentFeatures(
                rms=rms_val,
                zcr=zcr,
                centroid=centroid,
                flatness=flatness,
                rolloff=rolloff,
                beat_strength=beat_strength,
                beat_regularity=beat_regularity,
                voice_ratio=voice_ratio,
                pitch_strength=pitch_strength_norm,
                # RMS 百分位
                rms_p10=gs.get('rms_p10', 0.005),
                rms_p15=gs.get('rms_p15', 0.010),
                rms_p20=gs.get('rms_p20', 0.012),
                rms_p25=gs.get('rms_p25', 0.015),
                rms_p50=gs.get('rms_p50', 0.04),
                rms_p60=gs.get('rms_p60', 0.055),
                rms_p75=gs.get('rms_p75', 0.07),
                rms_p90=gs.get('rms_p90', 0.10),
                rms_mean=gs.get('rms_mean', 0.045),
                # 过零率
                zcr_mean=gs.get('zcr_mean', 0.06),
                zcr_p60=gs.get('zcr_p60', 0.08),
                zcr_p75=gs.get('zcr_p75', 0.10),
                # 频谱质心
                centroid_mean=gs.get('centroid_mean', 2000),
                centroid_p50=gs.get('centroid_p50', 2000),
                centroid_p60=gs.get('centroid_p60', 2500),
                centroid_p75=gs.get('centroid_p75', 3000),
                # 平坦度
                flatness_mean=gs.get('flatness_mean', 0.05),
                flatness_p60=gs.get('flatness_p60', 0.07),
                flatness_p75=gs.get('flatness_p75', 0.09),
                # 节拍
                beat_mean=gs.get('beat_strength_mean', 0.3),
                # 滚降
                rolloff_p75=gs.get('rolloff_p75', 5000),
                rolloff_p90=gs.get('rolloff_p90', 6000),
                # 能量标准差
                rms_std=gs.get('rms_std', 0.03),
                rms_p95=gs.get('rms_p95', 0.12),
                # 节拍强度标准差
                beat_std=gs.get('beat_strength_std', 0.1),
                # v5.0 新增特征
                chroma_entropy=chroma_entropy_val,
                spectral_flux=spectral_flux_val,
                hf_zcr=hf_zcr_val,
                spectral_spread=spectral_spread_val,
                delta_mfcc_energy=delta_mfcc_energy_val,
                silence_ratio=silence_ratio_val,
                # v6.0 新增特征
                rms_variance=rms_variance_val,
                hf_energy_ratio=hf_energy_ratio_val,
                harmonic_ratio=harmonic_ratio_val,
                tempo=tempo_val,
                # 歌曲内相对能量基准（优化1）
                song_rms_baseline=ctx.get('song_rms_baseline', gs.get('rms_mean', 0.045)),
                relative_energy=ctx.get('relative_energy', 1.0),
                song_rms_p50=ctx.get('song_rms_p50', gs.get('rms_p50', 0.04)),
                song_rms_p75=ctx.get('song_rms_p75', gs.get('rms_p75', 0.07)),
                song_rms_p85=ctx.get('song_rms_p85', gs.get('rms_p90', 0.10)),
                # 频谱质心稳定性（优化3）
                centroid_stability=ctx.get('centroid_stability', 0.5),
                centroid_gradient=ctx.get('centroid_gradient', 0.0),
                centroid_relative=ctx.get('centroid_relative', 1.0),
                # BPM 稳态（优化3副）
                bpm_stability=ctx.get('bpm_stability', 0.5),
                # SSM 副歌似然（优化2）
                ssm_chorus_likelihood=ctx.get('ssm_chorus_likelihood', 0.0),
                # v7.1：结构特征（重复性）
                repeat_count=int(ctx.get('repeat_count', 0)),
                avg_similarity=float(ctx.get('avg_similarity', 0.0)),
            )

            # ── 分类逻辑（v6.0: 两阶段 + 位置上下文）─────────────────────────
            scores = self._score_all(features, context)

            best_label = max(scores, key=scores.get)

            # ── 置信度计算（v4.0 简化）──────────────────────────────────────────
            all_scores_sorted = sorted(scores.values(), reverse=True)
            if len(all_scores_sorted) > 1 and all_scores_sorted[0] > 0:
                gap = (all_scores_sorted[0] - all_scores_sorted[1]) / (all_scores_sorted[0] + 0.01)
                confidence = 0.5 + min(0.45, gap * 0.5)
            else:
                confidence = 0.8
            confidence = max(0.4, min(0.98, confidence))

            return best_label, confidence

        except Exception as e:
            logger.warning(f"分类失败: {e}")
            return LABEL_OTHER, 0.4

    # ── 私有工具 ──────────────────────────────────────────────────────────────

    def _get_voice_ratio(
        self,
        y: np.ndarray,
        sr: int,
        voice_mask_fps: Optional[Tuple[np.ndarray, float]],
        vocals_y: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算该片段内人声占比。

        优先级：
          1. Demucs 人声轨能量比（最准确）
          2. Silero/librosa VAD 掩码
          3. 规则估算（过零率 + 能量）
        """
        # A. Demucs 人声轨：用人声轨能量 / 全轨能量
        if vocals_y is not None and len(vocals_y) > 0:
            # 截断到相同长度
            n = min(len(y), len(vocals_y))
            rms_full   = float(np.sqrt(np.mean(y[:n] ** 2))) + 1e-8
            rms_vocals = float(np.sqrt(np.mean(vocals_y[:n] ** 2)))
            ratio = min(1.0, rms_vocals / rms_full)
            return ratio

        # VAD 掩码
        if voice_mask_fps is not None:
            mask, fps = voice_mask_fps
            if len(mask) > 0:
                return float(mask.mean())

        # 规则估算
        rms_frames = librosa.feature.rms(y=y, hop_length=512)[0]
        zcr_frames = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
        voice_frames = (
            (zcr_frames > 0.02) & (zcr_frames < 0.18) &
            (rms_frames > np.percentile(rms_frames, 20))
        )
        return float(voice_frames.mean())

    # ══════════════════════════════════════════════════════════════════════════
    # 评分辅助函数（优化 v4.0）
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _in_range(value: float, low: float, high: float) -> bool:
        """判断值是否在范围内（不含边界）"""
        return low < value < high

    @staticmethod
    def _above(value: float, threshold: float) -> bool:
        """判断值是否高于阈值"""
        return value > threshold

    @staticmethod
    def _below(value: float, threshold: float) -> bool:
        """判断值是否低于阈值"""
        return value < threshold

    @staticmethod
    def _relative_above(value: float, base: float, factor: float) -> bool:
        """判断值是否高于基准值的 factor 倍"""
        return value > base * factor

    @staticmethod
    def _relative_below(value: float, base: float, factor: float) -> bool:
        """判断值是否低于基准值的 factor 倍"""
        return value < base * factor

    @staticmethod
    def _score(conditions: List[Tuple[bool, float]]) -> float:
        """
        批量评分：[(条件, 分数), ...]
        返回所有满足条件的分数之和
        """
        return sum(score for condition, score in conditions if condition)

    def _score_all(self, features: SegmentFeatures, context: Optional[Dict] = None) -> Dict[str, float]:
        """
        两阶段分类 v6.0

        Stage 1: 大类判断（互斥优先级）
          silence → speech → talk → crowd → audience → (进入 Stage 2)
        Stage 2: 音乐细分
          intro / interlude / solo / verse / chorus / outro

        Args:
            features: 当前段落特征
            context: 上下文信息，包含 {
                'position_ratio': 在歌曲中的相对位置 (0.0~1.0),
                'prev_label': 前一段标签,
                'next_label': 后一段标签（可能为 None）,
                'prev_rms': 前一段 RMS,
                'next_rms': 后一段 RMS,
                'segment_index': 当前段落索引,
                'all_segment_count': 段落总数,
            }
        """
        f = features
        ctx = context or {}

        # 调试：打印第一个段落的特征
        segment_index = ctx.get('segment_index', 0)
        if segment_index == 0:
            logger.debug(
                f"首个段落特征: "
                f"rms={f.rms:.4f} rms_mean={f.rms_mean:.4f} rms_p15={f.rms_p15:.4f} "
                f"beat_regularity={f.beat_regularity:.4f} voice_ratio={f.voice_ratio:.4f} "
                f"harmonic_ratio={f.harmonic_ratio:.4f} silence_ratio={f.silence_ratio:.4f} "
                f"flatness={f.flatness:.4f} flatness_mean={f.flatness_mean:.4f} "
                f"chroma_entropy={f.chroma_entropy:.4f} spectral_flux={f.spectral_flux:.4f}"
            )

        # ══════════════════════════════════════════════════════════════════
        # Stage 1: 大类判断（非音乐优先检出）
        # ══════════════════════════════════════════════════════════════════

        # ── 1. 静默 ─────────────────────────────────────────────────────────
        if f.silence_ratio > 0.7 or f.rms < 1e-4:
            return {LABEL_SILENCE: 10.0}

        # ── 2. 讲话（纯讲话，无伴奏）────────────────────────────────────────
        speech_score = self._score([
            (f.beat_regularity < 0.15, 4.0),
            (f.beat_regularity < 0.25, 2.0),
            (f.silence_ratio > 0.35, 3.5),
            (f.silence_ratio > 0.50, 2.0),
            (self._in_range(f.voice_ratio, 0.20, 0.60), 3.0),
            (f.voice_ratio < 0.20, 1.0),
            (f.rms < f.rms_p15, 3.0),
            (self._in_range(f.rms, f.rms_p10, f.rms_p25), 2.0),
            (f.spectral_flux < 2.5, 2.0),
            (f.delta_mfcc_energy < 1.5, 1.5),
            (f.harmonic_ratio < 0.3, 2.0),
            (f.chroma_entropy > 2.5, 1.0),
        ])
        speech_score -= self._score([
            (f.beat_regularity > 0.5, 5.0),
            (f.rms > f.rms_p75, 4.0),
            (f.chroma_entropy < 1.5, 3.0),
            # 唱歌排除：高节奏规律性 + 高谐波比 + 高人声 → 在唱歌
            (f.beat_regularity > 0.35 and f.harmonic_ratio > 0.5 and f.voice_ratio > 0.25, 8.0),
            (f.beat_regularity > 0.35 and f.harmonic_ratio > 0.6, 6.0),
            # 强唱歌排除：人声占比高 + 谐波比高 → 演唱会中必然是唱歌
            (f.voice_ratio > 0.7 and f.harmonic_ratio > 0.7, 12.0),
            (f.voice_ratio > 0.8 and f.harmonic_ratio > 0.6, 10.0),
            # 关键修复：有人声 + 高谐波 → 在唱歌，即使 beat_regularity 低（慢歌/抒情段落）
            # 但如果 beat_regularity 极低（< 0.1）且 silence_ratio 高（> 0.3），则是讲话，不减分
            (f.voice_ratio > 0.3 and f.harmonic_ratio > 0.6 and not (f.beat_regularity < 0.1 and f.silence_ratio > 0.3), 8.0),
            (f.voice_ratio > 0.25 and f.harmonic_ratio > 0.7 and not (f.beat_regularity < 0.1 and f.silence_ratio > 0.3), 10.0),
        ])
        if speech_score >= 6.0:
            logger.debug(f"Stage 1: speech_score={speech_score:.2f} >= 6.0, 分类为 {LABEL_SPEECH if not (f.harmonic_ratio > 0.3 and f.beat_regularity > 0.15) else LABEL_TALK}")
            if f.harmonic_ratio > 0.3 and f.beat_regularity > 0.15:
                return {LABEL_TALK: speech_score}
            return {LABEL_SPEECH: speech_score}

        # ── 3. 暖场互动（talk: 有低音量伴奏的讲话）─────────────────────────
        # 互动必须有一定人声（voice_ratio > 0.15），否则是纯音乐
        # 同时 harmonic_ratio 不能太高（高谐波 = 职业演唱，不是互动讲话）
        if f.voice_ratio > 0.15 and f.harmonic_ratio < 0.75:
            talk_score = self._score([
                (f.silence_ratio > 0.25, 3.0),
                (self._in_range(f.beat_regularity, 0.15, 0.40), 3.0),
                (self._in_range(f.voice_ratio, 0.20, 0.55), 2.5),
                (f.harmonic_ratio > 0.3, 2.0),
                (f.rms < f.rms_p50, 2.0),
                (f.spectral_flux < 4.0, 1.5),
                (f.chroma_entropy > 2.0, 1.0),
            ])
            talk_score -= self._score([
                (f.beat_regularity > 0.6, 4.0),
                (f.rms > f.rms_p75, 3.0),
                # 唱歌排除：高节奏规律性 + 高谐波比 → 在唱歌，不是互动
                (f.beat_regularity > 0.35 and f.harmonic_ratio > 0.5, 8.0),
                (f.beat_regularity > 0.35 and f.harmonic_ratio > 0.6, 6.0),
                # 强唱歌排除：人声占比高 + 谐波比高 → 在唱歌
                (f.voice_ratio > 0.7 and f.harmonic_ratio > 0.7, 12.0),
                (f.voice_ratio > 0.8 and f.harmonic_ratio > 0.6, 10.0),
                # 关键修复：有人声 + 高谐波 → 在唱歌，即使 beat_regularity 低
                (f.voice_ratio > 0.3 and f.harmonic_ratio > 0.6, 8.0),
                (f.voice_ratio > 0.25 and f.harmonic_ratio > 0.7 and not (f.beat_regularity < 0.1 and f.silence_ratio > 0.3), 10.0),
                # 高 flatness → 噪声/合唱，不是讲话互动
                (f.flatness > f.flatness_mean * 2.0, 3.0),
                (f.hf_zcr > 0.12, 2.0),
            ])
            if talk_score >= 6.0:
                logger.debug(f"Stage 1: talk_score={talk_score:.2f} >= 6.0, 分类为 {LABEL_TALK}")
                return {LABEL_TALK: talk_score}
        else:
            talk_score = 0.0

        # ── 4. 掌声/欢呼（crowd: 纯噪声，无旋律）───────────────────────────
        # ZCR 辅助规则：高 ZCR (>0.15) 表明嘈杂/观众噪声
        crowd_score = self._score([
            (f.hf_energy_ratio > 0.35, 4.0),
            (f.hf_energy_ratio > 0.25, 2.0),
            (f.rms_variance > f.rms * 2.0, 3.5),
            (f.flatness > f.flatness_mean * 2.5, 3.0),
            (f.flatness > f.flatness_mean * 1.8, 1.5),
            (f.harmonic_ratio < 0.2, 3.0),
            (f.hf_zcr > 0.15, 3.0),  # 提高权重：高频 ZCR 是观众噪声的强特征
            (f.spectral_spread > 3500, 2.0),
            (f.beat_regularity < 0.3, 2.0),
            (f.chroma_entropy > 3.0, 1.5),
            (f.pitch_strength < 0.05, 1.5),
            # 新增：ZCR 辅助判断 - 高频 ZCR > 0.12 + 低谐波 = 观众
            (f.hf_zcr > 0.12 and f.harmonic_ratio < 0.5, 2.0),
        ])
        crowd_score -= self._score([
            (f.harmonic_ratio > 0.4, 5.0),
            (f.pitch_strength > 0.15, 3.0),
            (f.chroma_entropy < 2.0, 3.0),
        ])
        if crowd_score >= CONFIG.CROWD_SCORE_THRESHOLD:
            logger.debug(f"Stage 1: crowd_score={crowd_score:.2f} >= {CONFIG.CROWD_SCORE_THRESHOLD}, 分类为 {LABEL_CROWD}")
            return {LABEL_CROWD: crowd_score}

        # ── 5. 观众合唱（audience: 跟着旋律唱）─────────────────────────────
        # 观众合唱必须有一定人声（voice_ratio > 0.15）—— 演唱会优化：降低要求
        if f.voice_ratio > 0.15:
            audience_score = self._score([
                (f.flatness > f.flatness_mean * 1.5, 2.5),  # 演唱会优化：降低要求
                (f.hf_zcr > 0.12, 2.0),  # 演唱会优化：降低要求
                # ── 优化3：高频能量占比（>4kHz）—— 观众多人声叠加产生梳状滤波，宽频特征
                (f.hf_energy_ratio > 0.25, 3.0),  # 演唱会优化：降低要求
                (f.hf_energy_ratio > 0.15, 1.5),  # 演唱会优化：降低要求
                (self._in_range(f.voice_ratio, 0.15, 0.95), 2.0),  # 演唱会优化：扩大范围
                (f.beat_regularity < 0.75, 1.5),  # 演唱会优化：大幅提高
                (f.pitch_strength > 0.05, 2.0),  # 演唱会优化：降低要求
                (self._in_range(f.chroma_entropy, 1.2, 4.0), 1.5),  # 演唱会优化：扩大范围
                (f.harmonic_ratio > 0.15, 1.5),  # 演唱会优化：降低要求
            ])
            audience_score -= self._score([
                # 职业歌手演唱：voice_ratio 很高 + chroma_entropy 很低（音高稳定）
                (f.voice_ratio > 0.95 and f.chroma_entropy < 2.5, 4.0),  # 演唱会优化：提高阈值
                (f.voice_ratio > 0.90 and f.harmonic_ratio > 0.85, 3.0),  # 演唱会优化：提高阈值
                # 有明显 beat_regularity 且 voice_ratio 很高 -> 主流音乐演唱
                (f.beat_regularity > 0.5 and f.voice_ratio > 0.90, 3.0),  # 演唱会优化：提高阈值
                # beat_regularity 高 + harmonic_ratio 高 -> 职业演唱，不是观众合唱
                (f.beat_regularity > 0.6 and f.harmonic_ratio > 0.75, 4.0),  # 演唱会优化：提高阈值
                # 高能量 + 高谐波 -> 职业演唱
                (f.rms > f.rms_mean * 1.2 and f.harmonic_ratio > 0.8, 3.0),  # 演唱会优化：提高阈值
                # 关键：beat_regularity > 0.5 + harmonic_ratio > 0.8 -> 职业演唱（最强排除）
                (f.beat_regularity > 0.5 and f.harmonic_ratio > 0.8, 5.0),  # 演唱会优化：提高阈值
                # voice_ratio 中等偏高 + harmonic_ratio 高 -> 职业演唱
                (f.voice_ratio > 0.5 and f.harmonic_ratio > 0.85, 4.0),  # 演唱会优化：提高阈值
                # 关键修复：慢歌也适用 - 有人声 + 高谐波 -> 职业演唱
                # 但如果 hf_zcr > 0.12（高频噪声/观众特征），不减分
                (f.voice_ratio > 0.4 and f.harmonic_ratio > 0.8 and f.hf_zcr <= 0.12, 6.0),  # 演唱会优化：提高阈值
                (f.voice_ratio > 0.4 and f.harmonic_ratio > 0.8 and f.hf_zcr > 0.12, 0.0),  # 豁免
                # ── 优化3：质心稳定性排除——主歌音色稳定，患者合唱质心波动大
                # centroid_stability 高（>0.7）且 harmonic_ratio 高 → 职业歌手稳定演唱
                (f.centroid_stability > 0.75 and f.harmonic_ratio > 0.7, 4.0),  # 演唱会优化：提高阈值
                # 质心相对稳定 + 中高能量 → 职业歌手，主歌
                (f.centroid_relative > 0.8 and f.centroid_relative < 1.3 and f.harmonic_ratio > 0.7, 3.0),  # 演唱会优化：提高阈值
                # 高频能量低（<0.15）+ 谐波比高 → 职业歌手清晰人声，不是观众嘈杂
                (f.hf_energy_ratio < 0.15 and f.harmonic_ratio > 0.75, 5.0),  # 演唱会优化：提高阈值
            ])
            if audience_score >= CONFIG.AUDIENCE_SCORE_THRESHOLD:
                logger.debug(f"Stage 1: audience_score={audience_score:.2f} >= {CONFIG.AUDIENCE_SCORE_THRESHOLD}, 分类为 {LABEL_AUDIENCE}")
                return {LABEL_AUDIENCE: audience_score}
        else:
            audience_score = 0.0

        # ══════════════════════════════════════════════════════════════════
        # Stage 2: 音乐细分（v7.1 双轨融合：结构特征 + 声学特征）
        # ══════════════════════════════════════════════════════════════════
        # 核心思路：
        #   1. 结构特征（repeat_count, avg_similarity, position_ratio）→
        #      决定段落在大局中的"角色"（重复段→chorus候选，非重复段→verse等）
        #   2. 声学特征（energy, voice_ratio, harmonic_ratio 等）→
        #      辅助确认/微调（声学像chorus但结构说verse→可能升级）
        #   3. 两者矛盾时取加权决断
        # ══════════════════════════════════════════════════════════════════

        relative_rms = f.rms / (f.rms_mean + 1e-8)
        position_ratio = ctx.get('position_ratio', 0.5)
        prev_label = ctx.get('prev_label', None)
        next_label = ctx.get('next_label', None)
        prev_rms = ctx.get('prev_rms', None)
        next_rms = ctx.get('next_rms', None)

        # ── A. 结构特征分析 ────────────────────────────────────────────
        is_repeated = f.repeat_count >= 1 and f.avg_similarity > 0.75
        is_unique = not is_repeated

        # ── B. 声学特征快速评估 ────────────────────────────────────────
        song_rel_energy = f.relative_energy
        has_strong_voice = f.voice_ratio > 0.30
        is_high_energy = song_rel_energy > 1.05
        is_low_energy = song_rel_energy < 0.70
        is_pure_music = f.voice_ratio < 0.15

        # ══════════════════════════════════════════════════════════════════
        # ── 分类决策树 ──────────────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════

        # ── 规则1：纯音乐段落（无人声）────────────────────────────────
        if is_pure_music:
            if is_repeated:
                # 重复的纯音乐 → 间奏
                scores = self._build_acoustic_scores(f, ctx)
                scores[LABEL_INTERLUDE] += 8.0  # 结构强力加分
                scores[LABEL_CHORUS] -= 6.0
                scores[LABEL_VERSE] -= 4.0
                best_label = max(scores, key=scores.get)
                best_score = scores[best_label]
                logger.debug(
                    f"Stage 2: [规则1-纯音乐+重复] → {best_label} "
                    f"(repeat={f.repeat_count}, sim={f.avg_similarity:.2f})"
                )
                return scores
            else:
                # 非重复纯音乐 → 按位置判断 intro/interlude/solo/outro
                scores = self._build_acoustic_scores(f, ctx)
                if position_ratio < 0.15:
                    scores[LABEL_INTRO] += 8.0
                elif position_ratio > 0.85:
                    scores[LABEL_OUTRO] += 8.0
                elif 0.20 < position_ratio < 0.80:
                    scores[LABEL_INTERLUDE] += 6.0
                    scores[LABEL_SOLO] += 4.0
                    # 高谐波比 → 有旋律的乐器独奏
                    if f.harmonic_ratio > 0.5:
                        scores[LABEL_SOLO] += 3.0
                        scores[LABEL_INTERLUDE] -= 2.0
                scores[LABEL_CHORUS] -= 8.0
                scores[LABEL_VERSE] -= 6.0
                best_label = max(scores, key=scores.get)
                best_score = scores[best_label]
                logger.debug(
                    f"Stage 2: [规则1-纯音乐+唯一] pos={position_ratio:.2f} → {best_label}"
                )
                return scores

        # ── 规则2：重复段落 + 有人声 → 大概率是副歌 ─────────────────
        if is_repeated and has_strong_voice:
            scores = self._build_acoustic_scores(f, ctx)

            # 结构特征：重复 + 有人声 → 强力倾向 chorus
            chorus_boost = 5.0 + min(f.repeat_count, 3) * 2.0  # 重复越多越确定
            chorus_boost += f.avg_similarity * 3.0              # 相似度越高越确定
            scores[LABEL_CHORUS] += chorus_boost

            # 声学辅助：能量高 → 确认 chorus
            if is_high_energy:
                scores[LABEL_CHORUS] += 3.0
            elif is_low_energy:
                # 重复但能量低 → 可能是慢歌副歌或情感副歌，保留 chorus 但降低
                scores[LABEL_CHORUS] -= 1.0
                # 如果能量真的很低（<0.5），可能是间奏中的哼唱
                if song_rel_energy < 0.50:
                    scores[LABEL_INTERLUDE] += 3.0

            # 抑制 verse：重复段不应该是 verse
            scores[LABEL_VERSE] -= 4.0

            # 位置修正（不强烈惩罚，因为有些歌一开头就唱副歌）
            if position_ratio > 0.90:
                scores[LABEL_OUTRO] += 3.0

            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]
            logger.debug(
                f"Stage 2: [规则2-重复+有人声] repeat={f.repeat_count} "
                f"sim={f.avg_similarity:.2f} energy={song_rel_energy:.2f} → {best_label}"
            )
            return scores

        # ── 规则3：重复段落 + 无人声（voice_ratio 0.15~0.30）─────────
        # 可能是间奏、独奏的重复出现
        if is_repeated and not has_strong_voice:
            scores = self._build_acoustic_scores(f, ctx)
            scores[LABEL_INTERLUDE] += 5.0
            if f.harmonic_ratio > 0.4 and f.pitch_strength > 0.10:
                scores[LABEL_SOLO] += 3.0
            scores[LABEL_VERSE] -= 3.0
            scores[LABEL_CHORUS] -= 3.0
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]
            logger.debug(
                f"Stage 2: [规则3-重复+弱人声] repeat={f.repeat_count} → {best_label}"
            )
            return scores

        # ── 规则4：非重复段落 + 有人声 → verse 为主 ─────────────────
        # 这是 verse 的主要来源
        scores = self._build_acoustic_scores(f, ctx)

        # 结构特征：非重复 + 有人声 → 强力倾向 verse
        scores[LABEL_VERSE] += 4.0

        # 声学辅助确认 verse
        if not is_high_energy:
            scores[LABEL_VERSE] += 2.0  # 中低能量 = 典型主歌

        # 声学特征与结构矛盾：非重复但声学特征很像副歌
        if is_high_energy and f.voice_ratio > 0.40 and f.harmonic_ratio > 0.5:
            # 可能是副歌的第一次出现（还没看到重复），或者独唱段落
            scores[LABEL_CHORUS] += 4.0  # 给 chorus 一个机会
            # 如果 SSM 也说这里是副歌区域，进一步确认
            scores[LABEL_CHORUS] += f.ssm_chorus_likelihood * CONFIG.SSM_FEATURE_WEIGHT

        # 抑制 chorus（非重复段落不太可能是副歌，除非声学特征非常强）
        scores[LABEL_CHORUS] -= 2.0

        # 位置修正
        if position_ratio < 0.15:
            scores[LABEL_INTRO] += 4.0
        elif position_ratio > 0.85:
            scores[LABEL_OUTRO] += 4.0

        # 前后能量对比
        if next_rms is not None and f.rms < next_rms * 0.85:
            scores[LABEL_VERSE] += 2.0  # 下一段更响 → 当前是铺垫（主歌）
        if prev_label == LABEL_VERSE:
            scores[LABEL_VERSE] += 1.0

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        logger.debug(
            f"Stage 2: [规则4-非重复+有人声] energy={song_rel_energy:.2f} "
            f"repeat={f.repeat_count} sim={f.avg_similarity:.2f} → {best_label}"
        )
        return scores

    def _build_acoustic_scores(
        self,
        f: 'SegmentFeatures',
        ctx: Dict,
    ) -> Dict[str, float]:
        """
        v7.1: 构建纯声学特征评分（不包含结构特征）。
        被 Stage 2 决策树调用，作为辅助评分基底。
        """
        relative_rms = f.rms / (f.rms_mean + 1e-8)
        position_ratio = ctx.get('position_ratio', 0.5)
        prev_label = ctx.get('prev_label', None)
        next_rms = ctx.get('next_rms', None)
        song_rel_energy = f.relative_energy

        scores = {
            LABEL_INTRO:      0.0,
            LABEL_INTERLUDE:  0.0,
            LABEL_SOLO:       0.0,
            LABEL_VERSE:      0.0,
            LABEL_CHORUS:     0.0,
            LABEL_OUTRO:      0.0,
        }

        # ── 前奏声学特征 ──────────────────────────────────────────
        intro_score = self._score([
            (f.voice_ratio < 0.15, 5.0),
            (self._in_range(f.voice_ratio, 0.15, 0.25), 2.5),
            (f.beat_regularity > 0.5, 3.0),
            (f.harmonic_ratio > 0.5, 2.0),
            (f.chroma_entropy < 2.0, 2.0),
            (f.spectral_flux < 4.0, 1.5),
            (f.pitch_strength < 0.15, 1.5),
        ])
        if position_ratio < 0.20:
            intro_score += 3.0
        elif position_ratio < 0.30:
            intro_score += 1.5
        if relative_rms < 1.0:
            intro_score += 1.0
        if f.voice_ratio > 0.4:
            intro_score -= 4.0
        scores[LABEL_INTRO] = intro_score

        # ── 间奏声学特征 ──────────────────────────────────────────
        interlude_score = self._score([
            (f.voice_ratio < 0.15, 4.0),
            (self._in_range(f.voice_ratio, 0.15, 0.25), 2.0),
            (f.beat_regularity > 0.4, 2.5),
            (f.harmonic_ratio > 0.4, 2.0),
            (f.chroma_entropy < 2.5, 1.5),
        ])
        if 0.25 < position_ratio < 0.80:
            interlude_score += 3.0
        elif 0.15 < position_ratio < 0.90:
            interlude_score += 1.0
        if position_ratio < 0.15:
            interlude_score -= 3.0
        if position_ratio > 0.85:
            interlude_score -= 3.0
        if f.voice_ratio > 0.4:
            interlude_score -= 4.0
        scores[LABEL_INTERLUDE] = interlude_score

        # ── 乐器独奏声学特征 ──────────────────────────────────────
        solo_score = self._score([
            (f.voice_ratio < 0.15, 3.5),
            (self._in_range(f.voice_ratio, 0.15, 0.25), 1.5),
            (f.pitch_strength > 0.15, 3.0),
            (f.harmonic_ratio > 0.5, 2.0),
            (f.centroid > f.centroid_mean, 1.5),
            (self._in_range(f.chroma_entropy, 1.0, 2.5), 1.5),
        ])
        if 0.20 < position_ratio < 0.85:
            solo_score += 2.5
        if f.rms_variance > f.rms * 0.5:
            solo_score += 1.0
        if f.voice_ratio > 0.35:
            solo_score -= 4.0
        scores[LABEL_SOLO] = solo_score

        # ── 副歌声学特征 ──────────────────────────────────────────
        chorus_score = 0.0

        # 相对能量（>=1.0 才有资格作为副歌，>1.0 有额外加分）
        if song_rel_energy >= 1.0:
            chorus_score += 3.0
            if song_rel_energy > 1.0:
                chorus_score += min((song_rel_energy - 1.0) * 10.0, 5.0)
        elif song_rel_energy < 0.8:
            chorus_score -= 2.0

        # SSM 副歌似然
        chorus_score += f.ssm_chorus_likelihood * CONFIG.SSM_FEATURE_WEIGHT

        # BPM 稳态 + 质心稳定
        chorus_score += f.bpm_stability * 2.0
        chorus_score += f.centroid_stability * 1.5

        # 声学特征
        chorus_score += self._score([
            (f.has_voice, 2.0),
            (f.voice_ratio > 0.35, 1.5),
            (f.harmonic_ratio > 0.4, 1.5),
            (f.chroma_entropy < 2.5, 1.0),
            (f.spectral_flux > 4.0, 1.0),
            (f.delta_mfcc_energy > 2.5, 1.0),
            (f.spectral_spread > 2500, 1.0),
        ])

        # 纯音乐不应该是副歌
        if f.voice_ratio < 0.15:
            chorus_score -= 5.0

        # 位置（轻微惩罚，不强烈）
        if position_ratio < 0.20:
            chorus_score -= 2.0
        if position_ratio > 0.90:
            chorus_score -= 2.0

        # 前后能量对比
        if ctx.get('prev_rms') is not None and f.rms > ctx['prev_rms'] * 1.2:
            chorus_score += 2.0
        if prev_label == LABEL_VERSE:
            chorus_score += 1.0

        # 谐波比惩罚
        if f.harmonic_ratio < 0.2:
            chorus_score -= 2.0

        scores[LABEL_CHORUS] = chorus_score

        # ── 主歌声学特征 ──────────────────────────────────────────
        verse_score = 0.0

        # 相对能量（中低 = 典型主歌，高能量应该是副歌）
        if 0.50 <= song_rel_energy < 1.0:
            verse_score += 2.0
        elif song_rel_energy < 0.50:
            verse_score += 0.5  # 极低能量段落，可能是前奏/间奏
        # 高能量惩罚（副歌区域）
        if song_rel_energy >= 1.0:
            verse_score -= 3.0
        if song_rel_energy > 1.15:
            verse_score -= 2.0
        if song_rel_energy > 1.30:
            verse_score -= 2.0

        # 声学特征
        verse_score += self._score([
            (f.has_voice, 2.5),
            (f.beat_regularity > 0.35, 1.5),
            (f.harmonic_ratio > 0.3, 1.5),
            (self._in_range(f.chroma_entropy, 1.5, 3.5), 1.5),
            (f.voice_ratio > 0.2, 1.0),
        ])

        # BPM 稳态
        verse_score += f.bpm_stability * 0.5

        # 纯音乐不是主歌
        if f.voice_ratio < 0.15:
            verse_score -= 3.0
        if f.harmonic_ratio < 0.2:
            verse_score -= 2.0

        # 位置
        if position_ratio < 0.30:
            verse_score += 1.5
        elif position_ratio < 0.40:
            verse_score += 0.5

        # 前后能量
        if next_rms is not None and f.rms < next_rms * 0.85:
            verse_score += 2.0
        if prev_label == LABEL_VERSE:
            verse_score += 1.0

        scores[LABEL_VERSE] = max(0.0, verse_score)

        # ── 尾奏声学特征 ──────────────────────────────────────────
        outro_score = self._score([
            (f.voice_ratio < 0.25, 3.5),
            (self._in_range(f.voice_ratio, 0.20, 0.35), 1.5),
            (f.beat_regularity > 0.4, 2.0),
            (f.harmonic_ratio > 0.3, 1.5),
            (f.rms < f.rms_p60, 2.0),
            (f.spectral_flux < 3.0, 1.5),
            (f.silence_ratio > 0.15, 1.5),
        ])
        if position_ratio > 0.80:
            outro_score += 3.5
        elif position_ratio > 0.70:
            outro_score += 2.0
        if prev_label == LABEL_CHORUS:
            outro_score += 1.5
        if relative_rms > 1.3:
            outro_score -= 2.0
        if f.voice_ratio > 0.4:
            outro_score -= 3.0
        scores[LABEL_OUTRO] = outro_score

        return scores

class SegmentSplitter:
    """
    在单首歌曲内切割段落
    目标：每段 25~45 秒，优先在音乐结构变化点切割，节拍对齐
    """

    def __init__(self, sample_rate: int = 22050, hop_length: int = 1024):
        self.sr = sample_rate
        self.hop = hop_length

    def split(
        self,
        y: np.ndarray,
        start_offset: float,
        target_sec: float = 25.0,  # 降低目标时长
        min_sec: float = 15.0,     # 降低最小时长，确保产生更多段落
        max_sec: float = 35.0,
    ) -> List[Tuple[float, float]]:
        """
        将音频切割为若干段落

        Args:
            y: 音频数据
            start_offset: 该片段在原始音频中的起始时间（秒）
            target_sec: 目标段落时长
            min_sec: 最小段落时长
            max_sec: 最大段落时长

        Returns:
            [(start_sec, end_sec), ...] 绝对时间
        """
        sr  = self.sr
        hop = self.hop
        fps = sr / hop
        total_sec = len(y) / sr

        if total_sec < min_sec:
            return [(start_offset, start_offset + total_sec)]

        # 节拍点
        try:
            _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        except Exception:
            beat_times = np.array([])

        # 结构变化特征
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
            mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
            chroma_diff = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
            mfcc_diff   = np.linalg.norm(np.diff(mfcc,   axis=1), axis=0)
            chroma_diff /= (chroma_diff.max() + 1e-6)
            mfcc_diff   /= (mfcc_diff.max()   + 1e-6)
            struct_diff  = 0.5 * chroma_diff + 0.5 * mfcc_diff
        except Exception:
            rms = librosa.feature.rms(y=y, hop_length=hop)[0]
            struct_diff = np.abs(np.diff(rms))
            struct_diff /= (struct_diff.max() + 1e-6)

        # 能量包络切割（备用方案：如果结构变化不够明显）
        rms_smooth = librosa.feature.rms(y=y, hop_length=hop*4)[0]  # 更粗粒度
        rms_smooth = scipy.ndimage.uniform_filter1d(rms_smooth, size=5)

        total_frames = int(total_sec * fps)
        target_f = int(target_sec * fps)
        min_f    = int(min_sec    * fps)
        max_f    = int(max_sec    * fps)

        # 候选切割点（结构变化大的帧）
        threshold = np.percentile(struct_diff, 70)
        candidates = np.where(struct_diff > threshold)[0].tolist()

        # 如果候选点太少，降低阈值重试
        if len(candidates) < 3:
            threshold = np.percentile(struct_diff, 50)
            candidates = np.where(struct_diff > threshold)[0].tolist()

        # 能量包络备用：找局部最低点
        rms_frames = len(rms_smooth)
        rms_fps = sr / (hop * 4)
        try:
            energy_valleys = signal.argrelmin(rms_smooth, order=min(3, len(rms_smooth) - 1))[0] if len(rms_smooth) > 3 else np.array([])
        except (IndexError, ValueError):
            energy_valleys = np.array([])

        boundaries_f = [0]
        last = 0

        while last < total_frames - min_f:
            ideal = last + target_f
            s_min = last + min_f
            s_max = min(last + max_f, total_frames)

            if s_min >= total_frames:
                break

            in_range = [f for f in candidates if s_min <= f <= s_max]

            if in_range:
                best = min(in_range, key=lambda f: abs(f - ideal) / target_f - struct_diff[min(f, len(struct_diff)-1)] * 0.3)
            else:
                # 找不到结构变化点 → 用能量谷底
                s_min_t = s_min / fps
                s_max_t = s_min / fps
                valley_t = [(t, rms_smooth[int(t * rms_fps)]) for t in
                            energy_valleys if s_min_t <= t <= s_max_t]
                if valley_t:
                    best_t = min(valley_t, key=lambda x: x[1])[0]
                    best = int(best_t * fps)
                    best = max(s_min, min(best, s_max))
                else:
                    best = min(range(s_min, s_max + 1), key=lambda f: abs(f - ideal))

            # 对齐到最近节拍（±2秒）
            if len(beat_times) > 0:
                best_t = best / fps
                nearby = beat_times[np.abs(beat_times - best_t) < 2.0]
                if len(nearby) > 0:
                    nearest = nearby[np.argmin(np.abs(nearby - best_t))]
                    aligned = int(nearest * fps)
                    aligned = max(s_min, min(aligned, s_max))
                    best = aligned

            boundaries_f.append(best)
            last = best

        boundaries_f.append(total_frames)

        # 转换为绝对时间
        result = []
        for i in range(len(boundaries_f) - 1):
            t_start = boundaries_f[i]   / fps + start_offset
            t_end   = boundaries_f[i+1] / fps + start_offset
            if t_end - t_start >= 3.0:
                result.append((t_start, t_end))

        return result


# ── 主分析器 ──────────────────────────────────────────────────────────────────

class AudioAnalyzer:
    """
    音频分析器主类 v4.0 (SSM增强版)
    流程：
      1. 加载音频
      2. A. Demucs 人声分离（可选）
      3. D. 全局特征统计 → 动态归一化阈值
      4. 全局 VAD（人声活动检测）
      5. SSM 结构分析（自相似矩阵 → 副歌重复检测）
      6. 歌曲边界检测（掌声 + 能量 + Chroma）
      7. 每首歌内部段落切割
      8. 每段落分类（使用动态阈值 + 人声轨 + SSM融合）
      9. 后处理（合并过短、修正首尾标签）
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 1024,
        enable_vad: bool = True,
        enable_demucs: bool = True,      # A. Demucs 人声分离
        demucs_model: str = "htdemucs",  # A. Demucs 模型
        enable_ssm: bool = True,         # E. SSM 结构分析
    ):
        self.sr = sample_rate
        self.hop = hop_length
        self.enable_vad = enable_vad
        self.enable_demucs = enable_demucs  # 懒检测：separate_vocals 内部会动态检查
        self.demucs_model = demucs_model
        self.enable_ssm = enable_ssm

        self.boundary_detector = SongBoundaryDetector(sample_rate, hop_length)
        self.splitter          = SegmentSplitter(sample_rate, hop_length)
        self.classifier        = SegmentClassifier(sample_rate)
        self.vad               = VoiceActivityDetector(sample_rate) if enable_vad else None
        # E. SSM 结构分析器（使用更小的 hop_length 以获得更精细的帧）
        self.structure_analyzer = StructureAnalyzer(sample_rate, hop_length // 2)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def analyze(
        self,
        audio_path: str,
        filename: Optional[str] = None,
        progress_callback=None,
    ) -> AnalysisResult:
        """
        完整音频分析入口 v3.0

        新增：
          D. analyze() 第一步先计算全局统计量，注入 SegmentClassifier
          A. 可选 Demucs 分离，人声轨用于分类，全轨用于边界检测
        """
        t0 = time.time()

        def _cb(cur, total, msg):
            if progress_callback:
                progress_callback(cur, total, msg)

        _cb(0.02, 1.0, "加载音频...")

        # 1. 加载
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        total_duration = len(y) / sr
        logger.info(f"音频加载完成，时长 {total_duration:.1f}s")

        # 1.5 音频降噪（使用 pedalboard NoiseGate）
        if PEDALBOARD_AVAILABLE and self.enable_vad:
            try:
                _cb(0.03, 1.0, "音频降噪...")
                # 转换为 float32 以兼容 pedalboard
                y = y.astype(np.float32)
                
                # 创建降噪效果器
                noise_gate = pedalboard.NoiseGate(
                    threshold_db=-40,      # 噪声门限 (dB)
                    ratio=4,                # 压缩比
                    attack_ms=1.0,         # 启动时间 (ms)
                    release_ms=100,        # 释放时间 (ms)
                )
                
                # 应用降噪（pedalboard 需要 (channels, samples) 格式）
                y_2d = np.stack([y, y])  # 转为双声道
                board = pedalboard.Pedalboard([noise_gate])
                y_denoised = board(y_2d, sr)
                y = y_denoised[0]  # 取第一声道
                
                logger.info("音频降噪完成")
            except Exception as e:
                logger.warning(f"降噪失败: {e}")

        # 2. 歌手信息
        singer = self.parse_singer_from_filename(filename or os.path.basename(audio_path))

        # ── D. 全局统计量（动态归一化）─────────────────────────────────────
        _cb(0.05, 1.0, "计算全局特征统计（动态归一化）...")
        global_stats = self._compute_global_stats(y)
        self.classifier.set_global_stats(global_stats)

        # ── A. Demucs 人声分离 ────────────────────────────────────────────
        _demucs_tmp_dir = None
        vocals_path: Optional[str] = None
        no_vocals_path: Optional[str] = None

        if self.enable_demucs:
            _cb(0.08, 1.0, "Demucs 人声分离（首次需下载模型，请稍候）...")
            _demucs_tmp_dir = tempfile.mkdtemp(prefix="demucs_")
            vocals_path, no_vocals_path = separate_vocals(
                audio_path,
                out_dir=_demucs_tmp_dir,
                model=self.demucs_model,
            )
            if vocals_path:
                logger.info("Demucs 人声分离成功，将用于段落分类")
            else:
                logger.info("Demucs 分离失败，降级到原始音频")

        # 加载人声轨（22050Hz 单声道，用于分类时的 voice_ratio）
        y_vocals: Optional[np.ndarray] = None
        if vocals_path and os.path.exists(vocals_path):
            try:
                y_vocals, _ = librosa.load(vocals_path, sr=self.sr, mono=True)
                logger.info(f"人声轨加载完成，时长 {len(y_vocals)/self.sr:.1f}s")
            except Exception as e:
                logger.warning(f"人声轨加载失败: {e}")

        # 3. 全局 VAD（使用原始混合音频）
        _cb(0.10, 1.0, "语音活动检测...")
        global_voice_mask = None
        global_vad_fps    = None
        if self.enable_vad and self.vad is not None:
            try:
                # A. 如果有 Demucs 人声轨，用它做 VAD，更精准
                vad_input = vocals_path if vocals_path else audio_path
                global_voice_mask, global_vad_fps = self.vad.get_voice_mask(vad_input)
                logger.info(f"VAD 完成，人声占比 {global_voice_mask.mean():.1%}")
            except Exception as e:
                logger.warning(f"VAD 失败: {e}")

        # E. SSM 结构分析（识别重复段落 → 副歌）
        ssm_result = None
        if self.enable_ssm:
            _cb(0.13, 1.0, "SSM 结构分析（识别副歌重复模式）...")
            try:
                ssm_result = self.structure_analyzer.analyze(
                    y,
                    progress_callback=lambda c, t, m: _cb(0.13 + 0.07 * c, 1.0, m)
                )
                logger.info(f"SSM 分析完成: {len(ssm_result.get('chorus_regions', []))} 个副歌候选区域")
            except MemoryError as e:
                # 内存不足，使用简化版能量包络检测
                logger.warning(f"SSM 内存不足，使用简化版能量包络检测: {e}")
                try:
                    chorus_regions = self.structure_analyzer.detect_chorus_by_energy_envelope(y, sr)
                    ssm_result = {
                        'chorus_regions': chorus_regions,
                        'n_frames': len(y) // (self.hop // 2),
                        'fps': self.sr / (self.hop // 2),
                    }
                except Exception as e2:
                    logger.warning(f"能量包络检测也失败: {e2}")
            except Exception as e:
                logger.warning(f"SSM 分析失败: {e}")
                ssm_result = None

        # 4. 歌曲边界检测（使用原始混合音频，保留鼓点等信息）
        _cb(0.15, 1.0, "检测歌曲边界...")
        boundaries = self.boundary_detector.detect_boundaries(
            y, total_duration,
            progress_callback=lambda c, t, m: _cb(0.15 + 0.25 * (c/t), 1.0, m)
        )
        song_boundaries = [0.0] + boundaries + [total_duration]
        logger.info(f"歌曲边界: {[f'{b:.1f}s' for b in song_boundaries]}")

        # 5. 逐首歌处理
        _cb(0.40, 1.0, "切割并分类段落...")
        songs: List[SongInfo] = []

        for song_idx in range(len(song_boundaries) - 1):
            song_start = song_boundaries[song_idx]
            song_end   = song_boundaries[song_idx + 1]
            song_dur   = song_end - song_start

            _cb(
                0.40 + 0.50 * (song_idx / max(1, len(song_boundaries) - 1)),
                1.0,
                f"处理第 {song_idx+1} 首歌 ({song_start:.0f}s-{song_end:.0f}s)..."
            )

            s_frame = int(song_start * sr)
            e_frame = int(song_end   * sr)
            y_song  = y[s_frame:e_frame]

            # ═══════════════════════════════════════════════════════════════
            # 优化1+2+3：预计算歌曲级特征统计量，供段落分类使用
            # ═══════════════════════════════════════════════════════════════
            # ── 优化1：歌曲内 RMS 统计 ─────────────────────────────────────
            song_rms_mean = 0.01
            song_rms_p50 = 0.01
            song_rms_p75 = 0.01
            song_rms_p85 = 0.01
            song_rms_frames = None
            song_rms_smooth = None
            try:
                song_rms_frames = librosa.feature.rms(y=y_song, hop_length=self.hop)[0]
                song_rms_smooth = scipy.ndimage.uniform_filter1d(song_rms_frames, size=int(sr / self.hop * 3))
                song_rms_mean = float(np.mean(song_rms_smooth))
                song_rms_p50  = float(np.percentile(song_rms_smooth, 50))
                song_rms_p75  = float(np.percentile(song_rms_smooth, 75))
                song_rms_p85  = float(np.percentile(song_rms_smooth, 85))
                # 更新分类器歌曲级字段（合并模式，不覆盖全局 rms_mean）
                self.classifier.set_global_stats({
                    'song_rms_mean': song_rms_mean,
                    'song_rms_p50':  song_rms_p50,
                    'song_rms_p75':  song_rms_p75,
                    'song_rms_p85':  song_rms_p85,
                })
            except Exception as _e:
                logger.debug(f"歌曲RMS统计计算失败: {_e}")

            # ── 优化3：歌曲级频谱质心统计（用于段落质心相对值）────────────
            song_centroid_mean = 2000.0
            song_centroid_smooth = None
            try:
                song_centroid_frames = librosa.feature.spectral_centroid(
                    y=y_song, sr=sr, hop_length=self.hop)[0]
                song_centroid_smooth = scipy.ndimage.uniform_filter1d(
                    song_centroid_frames, size=int(sr / self.hop * 5))
                song_centroid_mean = float(np.mean(song_centroid_smooth))
            except Exception as _e:
                logger.debug(f"歌曲质心统计计算失败: {_e}")

            # ── 优化2：SSM 副歌似然逐帧计算（高效近似：帧内相似度均值）────
            # 对每帧，计算其与前后 5 秒窗口内所有帧的余弦相似度均值
            song_ssm_likelihood = np.zeros(len(song_rms_frames) if song_rms_frames is not None else 1)
            try:
                fps_song = sr / self.hop
                win_frames_ssm = max(1, int(5 * fps_song))
                mfcc_song = librosa.feature.mfcc(y=y_song, sr=sr, n_mfcc=13, hop_length=self.hop)
                chroma_song = librosa.feature.chroma_cqt(y=y_song, sr=sr, hop_length=self.hop)
                feat_ssm = np.vstack([mfcc_song, chroma_song]).astype(np.float32)
                norms_ssm = np.linalg.norm(feat_ssm, axis=0, keepdims=True)
                norms_ssm = np.maximum(norms_ssm, 1e-8)
                feat_ssm = feat_ssm / norms_ssm
                n_ssm = feat_ssm.shape[1]
                ssm_like = np.zeros(n_ssm, dtype=np.float32)
                for j in range(n_ssm):
                    w0 = max(0, j - win_frames_ssm)
                    w1 = min(n_ssm, j + win_frames_ssm + 1)
                    window = feat_ssm[:, w0:w1]
                    sims = np.dot(feat_ssm[:, j], window)
                    ssm_like[j] = float(np.mean(sims))
                # 归一化到 0~1（原始值范围大致 0.5~1.0）
                ssm_like = np.clip((ssm_like - 0.5) * 2.0, 0.0, 1.0)
                song_ssm_likelihood = ssm_like
            except Exception as _e:
                logger.debug(f"歌曲内SSM副歌似然计算失败: {_e}")

            # ── 优化3：BPM 稳态（歌曲级）──────────────────────────────────
            song_bpm_stability = 0.5
            try:
                _, beat_frames_song = librosa.beat.beat_track(y=y_song, sr=sr, hop_length=self.hop)
                beat_times_song = librosa.frames_to_time(beat_frames_song, sr=sr, hop_length=self.hop)
                if len(beat_times_song) > 3:
                    intervals_song = np.diff(beat_times_song)
                    song_bpm_stability = float(np.clip(
                        1.0 - np.std(intervals_song) / (np.mean(intervals_song) + 1e-8), 0, 1))
            except Exception as _e:
                logger.debug(f"BPM稳态计算失败: {_e}")

            # A. 提取对应的人声轨片段
            y_song_vocals: Optional[np.ndarray] = None
            if y_vocals is not None:
                v_s = int(song_start * self.sr)
                v_e = int(song_end   * self.sr)
                v_s = max(0, v_s)
                v_e = min(len(y_vocals), v_e)
                if v_e > v_s:
                    y_song_vocals = y_vocals[v_s:v_e]

            # 段落切割
            seg_ranges = self.splitter.split(y_song, song_start)

            # ── v7.0: 两遍处理 ──────────────────────────────────────────────
            # Pass 1: 计算每段的 RMS 和歌曲级特征（优化1+2+3）
            fps_song = sr / self.hop
            seg_rms_list = []
            seg_relative_energy_list = []   # 优化1: 歌曲内相对能量
            seg_centroid_stability_list = []  # 优化3: 质心稳定性
            seg_centroid_gradient_list = []   # 优化3: 质心梯度
            seg_centroid_relative_list = []   # 优化3: 质心相对值
            seg_ssm_likelihood_list = []      # 优化2: SSM 副歌似然

            for seg_start, seg_end in seg_ranges:
                sf = int((seg_start - song_start) * sr)
                ef = int((seg_end   - song_start) * sr)
                sf = max(0, sf)
                ef = min(len(y_song), ef)
                y_seg_p1 = y_song[sf:ef]
                seg_rms = float(np.sqrt(np.mean(y_seg_p1 ** 2))) if len(y_seg_p1) > 0 else 0.0
                seg_rms_list.append(seg_rms)

                # ── 优化1：歌曲内相对能量 ──────────────────────────────────
                seg_rel_e = seg_rms / max(song_rms_mean, 1e-8)
                seg_relative_energy_list.append(seg_rel_e)

                # ── 优化3：频谱质心稳定性 + 梯度 ─────────────────────────
                c_start_f = int((seg_start - song_start) * fps_song)
                c_end_f   = min(int((seg_end - song_start) * fps_song),
                                len(song_centroid_smooth) if song_centroid_smooth is not None else 0)
                if song_centroid_smooth is not None and c_end_f > c_start_f:
                    seg_centroid_frames_p1 = song_centroid_smooth[c_start_f:c_end_f]
                    seg_c_mean = float(np.mean(seg_centroid_frames_p1))
                    seg_c_std  = float(np.std(seg_centroid_frames_p1))
                    c_stab = 1.0 / (1.0 + seg_c_std / max(song_centroid_mean, 1))
                    c_grad = abs(float(seg_centroid_frames_p1[-1]) - float(seg_centroid_frames_p1[0])) / max(seg_end - seg_start, 1)
                    c_rel  = seg_c_mean / max(song_centroid_mean, 1)
                else:
                    c_stab, c_grad, c_rel = 0.5, 0.0, 1.0
                seg_centroid_stability_list.append(c_stab)
                seg_centroid_gradient_list.append(c_grad)
                seg_centroid_relative_list.append(c_rel)

                # ── 优化2：SSM 副歌似然均值 ──────────────────────────────
                if len(song_ssm_likelihood) > 1:
                    ssm_f0 = max(0, c_start_f)
                    ssm_f1 = min(len(song_ssm_likelihood), c_end_f)
                    if ssm_f1 > ssm_f0:
                        ssm_val = float(np.mean(song_ssm_likelihood[ssm_f0:ssm_f1]))
                    else:
                        ssm_val = 0.0
                else:
                    ssm_val = 0.0
                seg_ssm_likelihood_list.append(ssm_val)

            # ═══════════════════════════════════════════════════════════════
            # v7.1: 段落间 MFCC 相似度矩阵（结构特征核心）
            # ═══════════════════════════════════════════════════════════════
            # 对每段提取 MFCC 均值向量，然后计算两两余弦相似度
            seg_repeat_count_list = []
            seg_avg_similarity_list = []
            SIMILARITY_THRESHOLD = 0.85  # 相似度阈值
            try:
                seg_mfcc_vectors = []
                for seg_start, seg_end in seg_ranges:
                    sf_m = int((seg_start - song_start) * sr)
                    ef_m = int((seg_end - song_start) * sr)
                    sf_m = max(0, sf_m)
                    ef_m = min(len(y_song), ef_m)
                    y_seg_m = y_song[sf_m:ef_m]
                    if len(y_seg_m) > sr:  # 至少1秒才能提取MFCC
                        mfcc_seg = librosa.feature.mfcc(
                            y=y_seg_m, sr=sr, n_mfcc=13,
                            n_fft=min(2048, len(y_seg_m) // 4),
                            hop_length=min(512, len(y_seg_m) // 8)
                        )
                        mfcc_mean = mfcc_seg.mean(axis=1).astype(np.float32)
                    else:
                        mfcc_mean = np.zeros(13, dtype=np.float32)
                    seg_mfcc_vectors.append(mfcc_mean)

                n_segs = len(seg_ranges)
                norms = np.array([np.linalg.norm(v) for v in seg_mfcc_vectors])
                norms = np.maximum(norms, 1e-8)

                for i in range(n_segs):
                    sims_to_others = []
                    repeat = 0
                    vi = seg_mfcc_vectors[i] / norms[i]
                    for j in range(n_segs):
                        if i == j:
                            continue
                        vj = seg_mfcc_vectors[j] / norms[j]
                        sim = float(np.dot(vi, vj))
                        sims_to_others.append(sim)
                        if sim > SIMILARITY_THRESHOLD:
                            repeat += 1
                    seg_repeat_count_list.append(repeat)
                    seg_avg_similarity_list.append(
                        float(np.mean(sims_to_others)) if sims_to_others else 0.0
                    )
                logger.debug(
                    f"[v7.1 结构特征] {n_segs}段相似度矩阵计算完成, "
                    f"repeat_counts={seg_repeat_count_list}"
                )
            except Exception as _e:
                logger.debug(f"[v7.1] 段落MFCC相似度计算失败: {_e}")
                seg_repeat_count_list = [0] * len(seg_ranges)
                seg_avg_similarity_list = [0.0] * len(seg_ranges)

            # Pass 2: 带完整歌曲级上下文分类
            segments: List[Segment] = []
            for seg_idx, (seg_start, seg_end) in enumerate(seg_ranges):
                sf = int((seg_start - song_start) * sr)
                ef = int((seg_end   - song_start) * sr)
                sf = max(0, sf)
                ef = min(len(y_song), ef)
                y_seg = y_song[sf:ef]

                # VAD 子掩码
                vad_sub = None
                if global_voice_mask is not None and global_vad_fps is not None:
                    f0 = int(seg_start * global_vad_fps)
                    f1 = int(seg_end   * global_vad_fps)
                    f0 = max(0, f0)
                    f1 = min(len(global_voice_mask), f1)
                    sub_mask = global_voice_mask[f0:f1]
                    vad_sub = (sub_mask, global_vad_fps)

                # A. 人声轨子片段
                vocals_seg: Optional[np.ndarray] = None
                if y_song_vocals is not None:
                    v_sf = int((seg_start - song_start) * self.sr)
                    v_ef = int((seg_end   - song_start) * self.sr)
                    v_sf = max(0, v_sf)
                    v_ef = min(len(y_song_vocals), v_ef)
                    if v_ef > v_sf:
                        vocals_seg = y_song_vocals[v_sf:v_ef]

                # v7.0: 位置上下文 + 歌曲级特征（优化1+2+3）
                position_ratio = (seg_start - song_start) / song_dur if song_dur > 0 else 0.5
                seg_context = {
                    'position_ratio':     position_ratio,
                    'segment_index':      seg_idx,
                    'all_segment_count':  len(seg_ranges),
                    'prev_label':         segments[-1].label if segments else None,
                    'next_label':         None,  # 后续可能更新
                    'prev_rms':           seg_rms_list[seg_idx - 1] if seg_idx > 0 else None,
                    'next_rms':           seg_rms_list[seg_idx + 1] if seg_idx < len(seg_ranges) - 1 else None,
                    # ── 优化1：歌曲内相对能量基准 ─────────────────────────
                    'song_rms_baseline':  song_rms_mean,
                    'relative_energy':    seg_relative_energy_list[seg_idx],
                    'song_rms_p50':       song_rms_p50,
                    'song_rms_p75':       song_rms_p75,
                    'song_rms_p85':       song_rms_p85,
                    # ── 优化3：质心稳定性 + 梯度 ──────────────────────────
                    'centroid_stability': seg_centroid_stability_list[seg_idx],
                    'centroid_gradient':  seg_centroid_gradient_list[seg_idx],
                    'centroid_relative':  seg_centroid_relative_list[seg_idx],
                    # ── 优化3副：BPM 稳态（歌曲级）───────────────────────
                    'bpm_stability':      song_bpm_stability,
                    # ── 优化2：SSM 副歌似然 ────────────────────────────────
                    'ssm_chorus_likelihood': seg_ssm_likelihood_list[seg_idx],
                    # ── v7.1：结构特征（重复性）────────────────────────────
                    'repeat_count':    seg_repeat_count_list[seg_idx],
                    'avg_similarity':  seg_avg_similarity_list[seg_idx],
                }

                # D. 分类时传入动态阈值 + 完整歌曲级上下文
                label, conf = self.classifier.classify(
                    y_seg, vad_sub, vocals_y=vocals_seg, context=seg_context
                )

                segments.append(Segment(
                    start_time=seg_start,
                    end_time=seg_end,
                    label=label,
                    confidence=conf,
                    song_index=song_idx,
                    features={'position_ratio': position_ratio},
                ))

            segments = self._postprocess_segments(segments, y_song, song_start, sr)

            # E. SSM 融合：基于重复性重新分类（提升副歌识别准确率）
            if ssm_result is not None:
                segments = self.structure_analyzer.merge_with_classification(ssm_result, segments)

            if not segments:
                continue

            # BPM 估计
            bpm = None
            if song_dur > 30:
                try:
                    tempo, _ = librosa.beat.beat_track(y=y_song, sr=sr)
                    bpm = float(tempo)
                except Exception:
                    pass

            song = SongInfo(
                song_index=song_idx,
                song_name=f"Song_{song_idx + 1:02d}",
                segments=segments,
                start_time=song_start,
                end_time=song_end,
                bpm=bpm,
            )
            songs.append(song)

        _cb(0.95, 1.0, "整理结果...")

        # 清理 Demucs 临时文件
        if _demucs_tmp_dir and os.path.exists(_demucs_tmp_dir):
            try:
                shutil.rmtree(_demucs_tmp_dir)
            except Exception:
                pass

        audio_info = {
            'sample_rate': sr,
            'duration': total_duration,
            'channels': 1,
            'bit_depth': 16,
            'vad_enabled': self.enable_vad and global_voice_mask is not None,
            'silero_vad': SILERO_AVAILABLE,
            'demucs_enabled': self.enable_demucs and vocals_path is not None,  # A.
            'demucs_model': self.demucs_model if self.enable_demucs else None,  # A.
            'ssm_enabled': self.enable_ssm,  # E. SSM 结构分析
            'ssm_chorus_regions': len(ssm_result.get('chorus_regions', [])) if ssm_result else 0,  # E.
        }

        result = AnalysisResult(
            singer=singer,
            songs=songs,
            total_duration=total_duration,
            audio_info=audio_info,
            analysis_time=time.time() - t0,
        )

        _cb(1.0, 1.0, f"分析完成，共 {len(songs)} 首歌，{sum(len(s.segments) for s in songs)} 个段落")
        logger.info(f"分析完成，耗时 {result.analysis_time:.1f}s")
        return result

    # ── D. 全局统计量计算 ─────────────────────────────────────────────────────

    def _compute_global_stats(self, y: np.ndarray) -> Dict[str, float]:
        """
        对整段音频（全轨）计算全局特征统计量，用于动态归一化。

        使用适中的 hop（2048）平衡速度和准确性。
        """
        hop = 2048  # 适中的 hop，平衡速度和准确性
        sr  = self.sr

        try:
            rms      = librosa.feature.rms(y=y, frame_length=hop*2, hop_length=hop)[0]
            zcr      = librosa.feature.zero_crossing_rate(y, frame_length=hop*2, hop_length=hop)[0]
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
            flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
            rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]

            # 节拍强度
            try:
                onset_env    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
                beat_str_mean = float(onset_env.mean())
                beat_str_std = float(onset_env.std())
            except Exception:
                beat_str_mean = 0.3
                beat_str_std = 0.1

            # 过滤静默帧（rms < 全局 p5）
            rms_p5 = float(np.percentile(rms, 5))
            active = rms > rms_p5

            rms_active      = rms[active]      if active.sum() > 10 else rms
            zcr_active      = zcr[active]      if active.sum() > 10 else zcr
            centroid_active = centroid[active]  if active.sum() > 10 else centroid
            flatness_active = flatness[active]  if active.sum() > 10 else flatness
            rolloff_active  = rolloff[active]   if active.sum() > 10 else rolloff

            # v5.0 新增：高频ZCR统计、Chroma熵统计、谱通量统计
            # 高频ZCR（>2kHz 频段）
            try:
                from scipy.signal import butter, filtfilt
                nyq = sr / 2
                freq_low = 2000
                b, a = butter(4, freq_low / nyq, btype='high')
                y_hf = filtfilt(b, a, y)
                hf_zcr_global = librosa.feature.zero_crossing_rate(y_hf, frame_length=hop*2, hop_length=hop)[0]
                hf_zcr_global = hf_zcr_global[active] if active.sum() > 10 else hf_zcr_global
                hf_zcr_mean = float(hf_zcr_global.mean())
                hf_zcr_p60 = float(np.percentile(hf_zcr_global, 60))
                hf_zcr_p75 = float(np.percentile(hf_zcr_global, 75))
            except Exception:
                hf_zcr_mean, hf_zcr_p60, hf_zcr_p75 = 0.15, 0.20, 0.25

            # Chroma 熵统计
            try:
                chroma_global = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=hop*2, hop_length=hop)
                chroma_entropy_frames = []
                for i in range(chroma_global.shape[1]):
                    ch = chroma_global[:, i]
                    ch_norm = ch / (ch.sum() + 1e-8)
                    ch_norm = ch_norm[ch_norm > 0.01]
                    if len(ch_norm) > 0:
                        ce = -np.sum(ch_norm * np.log2(ch_norm + 1e-10))
                        chroma_entropy_frames.append(ce)
                if chroma_entropy_frames:
                    chroma_entropy_frames = np.array(chroma_entropy_frames)
                    chroma_entropy_mean = float(chroma_entropy_frames.mean())
                    chroma_entropy_p50 = float(np.percentile(chroma_entropy_frames, 50))
                    chroma_entropy_p75 = float(np.percentile(chroma_entropy_frames, 75))
                else:
                    chroma_entropy_mean, chroma_entropy_p50, chroma_entropy_p75 = 2.5, 2.5, 3.0
            except Exception:
                chroma_entropy_mean, chroma_entropy_p50, chroma_entropy_p75 = 2.5, 2.5, 3.0

            # 谱通量统计
            try:
                spec = np.abs(librosa.stft(y, n_fft=hop*2, hop_length=hop))
                flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
                flux = scipy.ndimage.uniform_filter1d(flux, size=int(sr / hop * 2))
                flux_active = flux[active] if active.sum() > 10 else flux
                spectral_flux_mean = float(flux_active.mean())
                spectral_flux_p75 = float(np.percentile(flux_active, 75))
            except Exception:
                spectral_flux_mean, spectral_flux_p75 = 5.0, 8.0

            # 频谱扩展度统计
            try:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
                bw_active = spectral_bandwidth[active] if active.sum() > 10 else spectral_bandwidth
                bandwidth_mean = float(bw_active.mean())
                bandwidth_p75 = float(np.percentile(bw_active, 75))
            except Exception:
                bandwidth_mean, bandwidth_p75 = 2000.0, 3000.0

            stats = {
                # RMS 百分位
                'rms_p10':  float(np.percentile(rms_active, 10)),
                'rms_p15':  float(np.percentile(rms_active, 15)),
                'rms_p20':  float(np.percentile(rms_active, 20)),
                'rms_p25':  float(np.percentile(rms_active, 25)),
                'rms_p50':  float(np.percentile(rms_active, 50)),
                'rms_p60':  float(np.percentile(rms_active, 60)),
                'rms_p75':  float(np.percentile(rms_active, 75)),
                'rms_p90':  float(np.percentile(rms_active, 90)),
                'rms_p95':  float(np.percentile(rms_active, 95)),
                'rms_mean': float(rms_active.mean()),
                'rms_std':  float(rms_active.std()),
                # 过零率
                'zcr_mean': float(zcr_active.mean()),
                'zcr_p60':  float(np.percentile(zcr_active, 60)),
                'zcr_p75':  float(np.percentile(zcr_active, 75)),
                # 频谱质心
                'centroid_mean': float(centroid_active.mean()),
                'centroid_p50':  float(np.percentile(centroid_active, 50)),
                'centroid_p60':  float(np.percentile(centroid_active, 60)),
                'centroid_p75':  float(np.percentile(centroid_active, 75)),
                # 频谱平坦度
                'flatness_mean': float(flatness_active.mean()),
                'flatness_p60':  float(np.percentile(flatness_active, 60)),
                'flatness_p75':  float(np.percentile(flatness_active, 75)),
                # 频谱滚降
                'rolloff_p75':   float(np.percentile(rolloff_active, 75)),
                'rolloff_p90':   float(np.percentile(rolloff_active, 90)),
                # 节拍强度
                'beat_strength_mean': beat_str_mean,
                'beat_strength_std': beat_str_std,
                # v5.0 新增
                'hf_zcr_mean': hf_zcr_mean,
                'hf_zcr_p60': hf_zcr_p60,
                'hf_zcr_p75': hf_zcr_p75,
                'chroma_entropy_mean': chroma_entropy_mean,
                'chroma_entropy_p50': chroma_entropy_p50,
                'chroma_entropy_p75': chroma_entropy_p75,
                'spectral_flux_mean': spectral_flux_mean,
                'spectral_flux_p75': spectral_flux_p75,
                'bandwidth_mean': bandwidth_mean,
                'bandwidth_p75': bandwidth_p75,
            }
            return stats

        except Exception as e:
            logger.warning(f"全局统计量计算失败，使用默认值: {e}")
            return {
                'rms_p10': 0.005, 'rms_p25': 0.015, 'rms_p50': 0.04,
                'rms_p60': 0.055, 'rms_p75': 0.07, 'rms_p90': 0.10, 'rms_p95': 0.12,
                'rms_mean': 0.05, 'rms_std': 0.03,
                'zcr_mean': 0.06, 'zcr_p60': 0.08, 'zcr_p75': 0.10,
                'centroid_mean': 2000, 'centroid_p50': 2000, 'centroid_p60': 2500, 'centroid_p75': 3000,
                'flatness_mean': 0.05, 'flatness_p60': 0.07, 'flatness_p75': 0.09,
                'rolloff_p75': 5000, 'rolloff_p90': 6000,
                'beat_strength_mean': 0.3, 'beat_strength_std': 0.1,
                # v5.0 新增默认值
                'hf_zcr_mean': 0.15, 'hf_zcr_p60': 0.20, 'hf_zcr_p75': 0.25,
                'chroma_entropy_mean': 2.5, 'chroma_entropy_p50': 2.5, 'chroma_entropy_p75': 3.0,
                'spectral_flux_mean': 5.0, 'spectral_flux_p75': 8.0,
                'bandwidth_mean': 2000.0, 'bandwidth_p75': 3000.0,
            }

    # ── 兼容旧接口 ────────────────────────────────────────────────────────────

    def analyze_segments(self, audio_path: str, progress_callback=None) -> List[Segment]:
        """兼容旧接口：返回所有段落（不区分歌曲）"""
        result = self.analyze(audio_path, progress_callback=progress_callback)
        all_segs = []
        for song in result.songs:
            all_segs.extend(song.segments)
        return all_segs

    def detect_songs(self, audio_path: str, progress_callback=None) -> List[SongInfo]:
        """兼容旧接口"""
        result = self.analyze(audio_path, progress_callback=progress_callback)
        return result.songs

    def parse_singer_from_filename(self, filename: str) -> str:
        """从文件名解析歌手信息"""
        singer, _ = self.parse_singer_and_concert(filename)
        return singer

    def parse_singer_and_concert(self, filename: str) -> tuple:
        """
        从文件名解析歌手和演唱会信息
        支持格式：
        - 歌手 - 演唱会名.mp4 → (歌手, 演唱会名)
        - 歌手_演唱会名.mp4 → (歌手, 演唱会名)
        - 歌手.mp4 → (歌手, "")
        """
        import re
        basename = os.path.splitext(os.path.basename(filename))[0]

        # 优先匹配 "歌手 - 演唱会名" 或 "歌手_演唱会名"
        # 注意：歌手名可能包含连字符（如 A-Lin），所以优先匹配 "空格-空格" 模式
        patterns = [
            r'^(.+?)\s+[-–—]\s+(.+)$',   # 歌手 - 演唱会名（优先：空格-空格）
            r'^(.+?)\s*[-–—]\s*(.+)$',   # 歌手-演唱会名（次选：无空格）
            r'^(.+?)[-_](.+)$',          # 歌手_演唱会名
        ]
        for pattern in patterns:
            match = re.match(pattern, basename)
            if match:
                singer = match.group(1).strip()
                concert = match.group(2).strip()
                # 去除网站标记如 "[牛牛视听馆]niutv.taobao.com"
                concert = re.sub(r'\s*\[.*?\].*$', '', concert)
                # 去除可能的日期前缀如 "20230101 "
                concert = re.sub(r'^\d+\s*', '', concert)
                # 如果演唱会是纯数字或太短，不作为演唱会名
                if concert and len(concert) > 2 and not concert.isdigit():
                    return singer, concert
                return singer, ""

        return basename, ""

    # ── 后处理 ────────────────────────────────────────────────────────────────

    def _merge_segments(self, seg1: Segment, seg2: Segment, label: str = None) -> Segment:
        """合并两个相邻段落"""
        return Segment(
            start_time=seg1.start_time,
            end_time=seg2.end_time,
            label=label or seg1.label,
            confidence=(seg1.confidence + seg2.confidence) / 2,
            song_index=seg1.song_index,
            features={},
        )

    def _relabel(self, seg: Segment, new_label: str) -> Segment:
        """重新标记段落"""
        return Segment(
            start_time=seg.start_time,
            end_time=seg.end_time,
            label=new_label,
            confidence=seg.confidence,
            song_index=seg.song_index,
            features={},
        )

    def _postprocess_segments(
        self,
        segments: List[Segment],
        y_song: np.ndarray,
        song_start: float,
        sr: int,
    ) -> List[Segment]:
        """
        后处理 v4.0（优化版）：
        1. 合并过短段落
        2. 修正首尾标签
        3. 合并相邻同类段落
        """
        if not segments:
            return segments

        # 1. 合并过短段落（合并为单次遍历）
        merged = []
        for seg in segments:
            if merged and seg.duration < MIN_MERGE_DURATION:
                merged[-1] = self._merge_segments(merged[-1], seg)
            else:
                merged.append(seg)

        # 2. 修正首尾标签
        if len(merged) >= 2:
            # 首段
            first = merged[0]
            if first.label in (LABEL_VERSE, LABEL_CHORUS, LABEL_INTERLUDE, LABEL_SOLO) and first.duration < 25:
                merged[0] = self._relabel(first, LABEL_INTRO)
            # 尾段
            last = merged[-1]
            if last.label in (LABEL_VERSE, LABEL_CHORUS, LABEL_INTERLUDE, LABEL_SOLO) and last.duration < 25:
                merged[-1] = self._relabel(last, LABEL_OUTRO)

        # 3. 合并相邻同类段落（谨慎处理，避免把所有verse合并成一个）
        # 只在间隔小于0.5秒且标签为同类型时合并
        if not merged:
            return []
        final = [merged[0]]
        for seg in merged[1:]:
            prev = final[-1]
            gap = seg.start_time - prev.end_time
            # 只有当 gap 很小且确实是相同结构时才合并
            if gap < 0.5 and seg.label == prev.label:
                # 不合并，保留各自独立
                final.append(seg)
            else:
                final.append(seg)

        return final


# ── 兼容旧代码的别名 ──────────────────────────────────────────────────────────

class SegmentAnalyzer:
    """向后兼容的别名"""
    def __init__(self, sample_rate=22050, hop_length=1024):
        self._analyzer = AudioAnalyzer(sample_rate, hop_length)

    def detect_segments(self, audio_path, progress_callback=None):
        return self._analyzer.analyze_segments(audio_path, progress_callback)


class BeatDetector:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def detect_beats(self, audio_path, progress_callback=None):
        y, sr = librosa.load(audio_path, sr=self.sr)
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        return beats.tolist()

    def estimate_bpm(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)


class FeatureExtractor:
    def __init__(self, sample_rate=22050, hop_length=1024):
        self.sr = sample_rate
        self.hop = hop_length

    def extract_features(self, audio_path, segment=None):
        if segment:
            y, sr = librosa.load(audio_path, sr=self.sr,
                                  offset=segment.start_time, duration=segment.duration)
        else:
            y, sr = librosa.load(audio_path, sr=self.sr)
        return {
            'rms': float(librosa.feature.rms(y=y, hop_length=self.hop).mean()),
            'zcr': float(librosa.feature.zero_crossing_rate(y).mean()),
            'centroid': float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        }


# ── 示例 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = AudioAnalyzer()
    audio_file = "example.wav"
    if os.path.exists(audio_file):
        result = analyzer.analyze(audio_file, "周杰伦-演唱会.mp4")
        print(f"歌手: {result.singer}")
        print(f"歌曲数: {len(result.songs)}")
        for song in result.songs:
            print(f"\n  {song.song_name} ({song.start_time:.0f}s-{song.end_time:.0f}s, BPM={song.bpm})")
            for seg in song.segments:
                cn = LABEL_CN.get(seg.label, seg.label)
                print(f"    [{cn}] {seg.start_time:.1f}s-{seg.end_time:.1f}s  置信度:{seg.confidence:.0%}")
