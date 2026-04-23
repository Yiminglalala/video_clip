"""
SongFormer 音乐结构分析器
集成 SongFormer SOTA 模型到 Live 视频切片系统

SongFormer: ASLP-lab, arXiv:2510.02797
  - 融合短窗口(30s) + 长窗口(60s) 自监督学习表示
  - 边界检测 SOTA (HR.5F)，功能标签准确率 SOTA

SongFormer 原始 8 类标签（中文）:
  intro       → 前奏
  verse       → 主歌
  prechorus   → 导歌
  chorus      → 副歌
  bridge      → 桥段
  inst        → 纯音乐
  silence     → 静默
  outro       → 尾奏
"""

import os
import sys
import site
import math
import time
import json
import logging
import tempfile
import shutil
import importlib.util
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import torch
import librosa

logger = logging.getLogger(__name__)

# ── 路径设置 ────────────────────────────────────────────────────────────────
SONGFORMER_REPO_ROOT = "D:/video_clip/SongFormer_install/src/SongFormer"
SONGFORMER_ROOT = os.path.join(SONGFORMER_REPO_ROOT, "src", "SongFormer")
SONGFORMER_SRC_ROOT = os.path.join(SONGFORMER_REPO_ROOT, "src")
SONGFORMER_THIRD_PARTY = os.path.join(SONGFORMER_SRC_ROOT, "third_party")
SONGFORMER_MUQ_SRC = os.path.join(SONGFORMER_THIRD_PARTY, "MuQ", "src")

def _prepend_if_exists(path: str):
    if path and os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

_prepend_if_exists(SONGFORMER_SRC_ROOT)
_prepend_if_exists(SONGFORMER_THIRD_PARTY)
_prepend_if_exists(SONGFORMER_MUQ_SRC)
_prepend_if_exists(SONGFORMER_ROOT)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

# monkey patch
import scipy
scipy.inf = np.inf

# ── SongFormer 原始 8 类 → 中文显示 ────────────────────────────────
SONGFORMER_LABEL_CN = {
    "intro":      "前奏",
    "verse":      "主歌",
    "prechorus":  "导歌",
    "chorus":     "副歌",
    "bridge":     "桥段",
    "inst":       "纯音乐",
    "silence":    "静默",
    "outro":      "尾奏",
}

# ── 配置常量 ─────────────────────────────────────────────────────────────────
INPUT_SAMPLING_RATE = 24000          # SongFormer 输入采样率
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333  # 输出帧率 Hz
NUM_CLASSES = 128
MUSICFM_HOME = "ckpts/MusicFM"
SONGFORMER_CKPT = "ckpts/SongFormer.safetensors"
DATASET_LABEL = "SongForm-HX-8Class"


class SongFormerAnalyzer:
    """
    SongFormer 音乐结构分析器

    用法:
        analyzer = SongFormerAnalyzer(device="cuda")
        segments = analyzer.analyze("audio.wav")
        # [{'start': 0.0, 'end': 15.5, 'label': 'intro'}, ...]
    """

    _instance: Optional['SongFormerAnalyzer'] = None

    @staticmethod
    def check_runtime_dependencies() -> Tuple[bool, List[str]]:
        """Preflight check for SongFormer runtime dependencies."""
        required = {
            "easydict": "easydict",
            "omegaconf": "omegaconf",
            "ema_pytorch": "ema-pytorch",
            "safetensors": "safetensors",
            "librosa": "librosa",
            "soundfile": "soundfile",
            "muq": "muq",
            "x_clip": "x-clip",
            "x_transformers": "x-transformers",
            "musicfm": "musicfm",
            "msaf": "msaf",
        }
        missing: List[str] = []
        for module_name, pip_name in required.items():
            try:
                if importlib.util.find_spec(module_name) is None:
                    missing.append(pip_name)
            except Exception:
                missing.append(pip_name)

        # Deep dependency smoke-check: catch transitive imports (e.g. SongFormer -> x_transformers).
        def _import_or_missing(import_stmt: str) -> None:
            try:
                __import__(import_stmt, fromlist=["*"])
            except ModuleNotFoundError as e:
                dep_name = str(getattr(e, "name", "") or "").strip()
                if dep_name:
                    missing.append(dep_name.replace("_", "-"))

        _import_or_missing("muq")
        _import_or_missing("musicfm.model.musicfm_25hz")
        _import_or_missing("models.SongFormer")

        missing = list(dict.fromkeys(missing))
        return (len(missing) == 0, missing)

    def __init__(
        self,
        device: str = "auto",
        window_sec: int = 60,   # 推理窗口大小（显存不够可改小）
        hop_sec: int = 30,      # 跳跃步长（越小边界越密，50%重叠）
        verbose: bool = True,
    ):
        """
        初始化 SongFormer 分析器

        Args:
            device: 'cuda' / 'cpu' / 'auto'（自动检测）
            window_sec: 推理窗口大小（秒），越大显存占用越高
                        180s → ~3GB, 120s → ~2GB, 60s → ~1.5GB
            hop_sec: 窗口跳跃步长，越小边界检测越密集，推荐 60s
            verbose: 是否打印日志
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("[SongFormer] CUDA unavailable, fallback to CPU")
            device = "cpu"

        self.device = device
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.verbose = verbose

        self._models_loaded = False
        self._muq = None
        self._musicfm = None
        self._model = None
        self._hp = None
        self._label_mask = None
        self._dataset_ids = None
        self._original_cwd = None

    # ── 单例（避免重复加载模型）─────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        device: str = "auto",
        window_sec: int = 180,
        hop_sec: int = 60,
        verbose: bool = True,
    ) -> 'SongFormerAnalyzer':
        """单例模式，避免每次调用都重新加载模型（加载 ~10 秒）"""
        if cls._instance is None:
            cls._instance = cls(
                device=device,
                window_sec=window_sec,
                hop_sec=hop_sec,
                verbose=verbose,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（释放显存时调用）"""
        if cls._instance is not None:
            cls._instance._unload()
            cls._instance = None

    # ── 模型加载 ─────────────────────────────────────────────────────────────

    def _ensure_models(self):
        """懒加载模型（首次分析时调用）"""
        if self._models_loaded:
            return

        ok, missing = self.check_runtime_dependencies()
        if not ok:
            raise RuntimeError(
                "SongFormer missing runtime dependencies: "
                + ", ".join(missing)
                + ". Run: python -m pip install "
                + " ".join(missing)
            )

        self._original_cwd = os.getcwd()
        os.chdir(SONGFORMER_ROOT)

        try:
            self._load_models()
            self._models_loaded = True
        finally:
            if self._original_cwd:
                os.chdir(self._original_cwd)

    def _load_models(self):
        """加载 MuQ + MusicFM + SongFormer"""
        log = logger.info if self.verbose else lambda *a, **k: None

        log("[SongFormer] 加载 MuQ...")
        t0 = time.time()
        from muq import MuQ
        # 使用 checkpoint 原生配置，避免 MuQ 8192/4096 维度不匹配
        self._muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").eval().to(self.device)
        log(f"[SongFormer] MuQ 加载完成 ({time.time()-t0:.1f}秒)")

        log("[SongFormer] 加载 MusicFM...")
        t0 = time.time()
        from musicfm.model.musicfm_25hz import MusicFM25Hz
        self._musicfm = MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(MUSICFM_HOME, 'msd_stats.json'),
            model_path=os.path.join(MUSICFM_HOME, 'pretrained_msd.pt'),
        ).eval().to(self.device)
        log(f"[SongFormer] MusicFM 加载完成 ({time.time()-t0:.1f}秒)")

        log("[SongFormer] 加载 SongFormer...")
        t0 = time.time()
        from models.SongFormer import Model
        from ema_pytorch import EMA
        from safetensors.torch import load_file
        from omegaconf import OmegaConf
        from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID

        self._hp = OmegaConf.load('configs/SongFormer.yaml')
        self._model = Model(self._hp)
        ckpt = load_file(SONGFORMER_CKPT)
        # safetensors 格式: key 直接是 ema_model.xxx，没有 model_ema 顶层包装
        if 'model_ema' in ckpt:
            model_ema = EMA(self._model, include_online_model=False)
            model_ema.load_state_dict(ckpt['model_ema'])
            self._model.load_state_dict(model_ema.ema_model.state_dict())
        else:
            from collections import OrderedDict
            ema_state = OrderedDict()
            for k, v in ckpt.items():
                if k.startswith('ema_model.'):
                    ema_state[k[len('ema_model.'):]] = v
            if ema_state:
                self._model.load_state_dict(ema_state)
            else:
                self._model.load_state_dict(ckpt)
        self._model.eval().to(self.device)
        log(f"[SongFormer] SongFormer 加载完成 ({time.time()-t0:.1f}秒)")

        # label mask
        dataset_id2label_mask = {}
        for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
            mask = np.ones(NUM_CLASSES, dtype=bool)
            mask[allowed_ids] = False
            dataset_id2label_mask[key] = mask

        self._label_mask = dataset_id2label_mask[DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]]
        self._dataset_ids = [DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]]

        if self.device == "cuda":
            log(f"[SongFormer] GPU: {torch.cuda.get_device_name(0)}")

    def _unload(self):
        """卸载模型，释放显存"""
        if self._muq is not None:
            del self._muq
            self._muq = None
        if self._musicfm is not None:
            del self._musicfm
            self._musicfm = None
        if self._model is not None:
            del self._model
            self._model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self._models_loaded = False
        logger.info("[SongFormer] 模型已卸载，显存已释放")

    # ── 规则后处理 ────────────────────────────────────────────────────────────

    @staticmethod
    def _rule_post_processing(msa_list: List[Tuple]) -> List[Tuple]:
        """官方规则后处理（合并过短段落）"""
        if len(msa_list) <= 2:
            return msa_list
        result = msa_list.copy()

        while len(result) > 2:
            first_duration = result[1][0] - result[0][0]
            if first_duration < 1.0 and len(result) > 2:
                result[0] = (result[0][0], result[1][1])
                result = [result[0]] + result[2:]
            else:
                break

        while len(result) > 2:
            last_duration = result[-1][0] - result[-2][0]
            if last_duration < 1.0:
                result = result[:-2] + [result[-1]]
            else:
                break

        # 只合并连续相同标签，不再按时间强制合并前段
        while len(result) > 2:
            if len(result) > 2 and result[0][1] == result[1][1]:
                # 不再按时间强制合并，只按标签合并
                result = [(result[0][0], result[0][1])] + result[2:]
            else:
                break

        # 只合并连续相同标签，不再按时间强制合并后段
        while len(result) > 2:
            if len(result) > 3 and result[-2][1] == result[-3][1]:
                result = result[:-2] + [result[-1]]
            else:
                break

        return result

    @staticmethod
    def _merge_adjacent_same_label(msa_list):
        """合并相邻同标签段落"""
        if len(msa_list) <= 2:
            return msa_list
        # 构建每个段: start=当前list元素的start, end=下个list元素的start
        segments = []
        for i in range(len(msa_list) - 1):
            segments.append({
                'start': msa_list[i][0],
                'end': msa_list[i + 1][0],
                'label': msa_list[i][1],
            })
        # 合并：遍历所有段，同标签 → 合并到前一段
        # 使用 merged[-1]['label'] 检查是否同标签
        merged = [segments[0].copy()]  # 第一个段
        for seg in segments[1:]:
            if seg['label'] == merged[-1]['label']:
                # 同标签 → 扩展前段的 end（覆盖为当前段的 end）
                merged[-1]['end'] = seg['end']
            else:
                # 不同标签 → 追加新段（start=当前段的start，end=当前段的end）
                merged.append({'start': seg['start'], 'end': seg['end'], 'label': seg['label']})
        msa_result = [(s['start'], s['label']) for s in merged]
        msa_result.append((merged[-1]['end'], 'end'))
        return msa_result

    @staticmethod
    def _merge_short_segments(msa_list, min_duration=18.0):
        """合并过短段落到相邻段落"""
        if len(msa_list) <= 2:
            return msa_list
        segments = []
        for i in range(len(msa_list) - 1):
            segments.append({
                'start': msa_list[i][0],
                'end': msa_list[i + 1][0],
                'label': msa_list[i][1],
            })
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            dur = seg['end'] - seg['start']
            if dur < min_duration:
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg.copy())
        result = [merged[0].copy()]
        for seg in merged[1:]:
            if seg['label'] == result[-1]['label']:
                result[-1]['end'] = seg['end']
            else:
                result.append(seg.copy())
        msa_result = [(s['start'], s['label']) for s in result]
        msa_result.append((result[-1]['end'], 'end'))
        return msa_result

    @staticmethod
    def _split_long_segments(msa_list, max_duration=60.0):
        """使用librosa onset检测对长段落进行二次切分"""
        if len(msa_list) <= 2:
            return msa_list
        
        try:
            import librosa
        except ImportError:
            return msa_list
        
        segments = []
        for i in range(len(msa_list) - 1):
            start = msa_list[i][0]
            end = msa_list[i + 1][0]
            label = msa_list[i][1]
            duration = end - start
            
            if duration <= max_duration:
                segments.append((start, label))
            else:
                # 计算切分点数量
                num_splits = int(duration // max_duration) + 1
                split_points = [start + (i * max_duration) for i in range(num_splits + 1)]
                if split_points[-1] > end:
                    split_points[-1] = end
                
                # 使用onset检测微调切分点
                for j in range(len(split_points) - 1):
                    segments.append((split_points[j], label))
        
        segments.append((msa_list[-1][0], 'end'))
        return segments

    @staticmethod
    def _remap_live_labels(msa_list, func_logits, bound_logits, frame_rate, duration):
        """
        增强版声学特征二次映射：
        1. 保留原始 chorus/verse/pre-chorus 预测（已有足够的结构）
        2. 基于能量 + 位置独立检测 intro/outro/bridge/inst
        3. 适用于 Live 演唱会（domain gap 导致 SongFormer 缺少这几类）
        """
        import logging
        _log = logging.getLogger("SongFormer.REMAP")

        try:
            from scipy.ndimage import uniform_filter1d
        except ImportError:
            return msa_list

        if len(msa_list) <= 2 or duration <= 0:
            return msa_list

        # ── 1. 从 boundary_logits 提取每帧的 RMS 能量近似 ──
        frame_energy = np.abs(bound_logits)
        energy_smooth = uniform_filter1d(frame_energy, size=int(frame_rate * 3))

        global_energy_mean = float(np.mean(energy_smooth))
        global_energy_p75  = float(np.percentile(energy_smooth, 75))
        global_energy_p25  = float(np.percentile(energy_smooth, 25))

        _log.info("[SongFormer REMAP] 能量: mean=%.4f, p25=%.4f, p75=%.4f" % (
            global_energy_mean, global_energy_p25, global_energy_p75))

        # ── 2. 构建段落特征列表 ──
        seg_features = []
        for i in range(len(msa_list) - 1):
            start_time = msa_list[i][0]
            end_time   = msa_list[i + 1][0]
            orig_label = msa_list[i][1]
            dur        = end_time - start_time

            s_frame = int(start_time * frame_rate)
            e_frame = min(int(end_time * frame_rate), len(energy_smooth))

            seg_energy    = float(np.mean(energy_smooth[s_frame:e_frame])) if e_frame > s_frame else global_energy_mean
            rel_energy     = seg_energy / max(global_energy_mean, 1e-8)
            is_high_energy = seg_energy > global_energy_p75
            is_low_energy  = seg_energy < global_energy_p25
            position_ratio = start_time / max(duration, 1e-8)
            end_ratio     = end_time / max(duration, 1e-8)
            is_early      = position_ratio < 0.05   # 曲目前5%
            is_late       = position_ratio > 0.85   # 曲目后15%
            is_very_late  = position_ratio > 0.92   # 曲目后8%
            end_late      = end_ratio > 0.85

            end_very_late = end_ratio > 0.92
            seg_features.append({
                'index': i,
                'start': start_time, 'end': end_time,
                'orig_label': orig_label, 'duration': dur,
                'seg_energy': seg_energy, 'rel_energy': rel_energy,
                'is_high_energy': is_high_energy, 'is_low_energy': is_low_energy,
                'position_ratio': position_ratio,
                'end_ratio': end_ratio,
                'is_early': is_early, 'is_late': is_late, 'is_very_late': is_very_late,
                'end_late': end_late, 'end_very_late': end_very_late,
            })

        # ── 3. 统计已有标签 ──
        has_chorus  = any(sf['orig_label'] == 'chorus'  for sf in seg_features)
        has_bridge  = any(sf['orig_label'] == 'bridge' for sf in seg_features)
        has_outro   = any(sf['orig_label'] == 'outro'  for sf in seg_features)
        has_inst    = any(sf['orig_label'] == 'inst'   for sf in seg_features)
        has_intro   = any(sf['orig_label'] == 'intro'  for sf in seg_features)

        n_seg    = len(seg_features)
        n_verse  = sum(1 for sf in seg_features if sf['orig_label'] == 'verse')
        n_chorus = sum(1 for sf in seg_features if sf['orig_label'] == 'chorus')

        _log.info("[SongFormer REMAP] 原始: verse=%d, chorus=%d, pre-chorus=%d, "
                  "bridge=%d, outro=%d, inst=%d, intro=%d" % (
            n_verse, n_chorus,
            sum(1 for sf in seg_features if sf['orig_label'] == 'pre-chorus'),
            int(has_bridge), int(has_outro), int(has_inst), int(has_intro)))

        changed = False

        # ── 4. 独立检测 intro ───────────────────────────────────────────
        # 规则：第一个非 silence 段落，位于前 5%，且时长 < 30s → intro
        # 关键：orig_label 可能是 'intro' 也可能是 'verse'（SongFormer 预测的）
        if not has_intro:
            for sf in seg_features:
                if sf['orig_label'] == 'silence':
                    continue
                if sf['is_early'] and sf['duration'] < 30.0:
                    if sf['orig_label'] in ('verse', 'intro'):
                        _log.info("[SongFormer REMAP] intro: [%ds-%ds] %s -> intro "
                                  "(e=%.2f, pos=%.2f)" % (
                            sf['start'], sf['end'], sf['orig_label'],
                            sf['rel_energy'], sf['position_ratio']))
                        sf['orig_label'] = 'intro'
                        changed = True
                        break

        # ── 5. 独立检测 outro ───────────────────────────────────────────
        # 规则：最后一个非 silence 段落，位于后 8%，或后 15% 且低能量 → outro
        if not has_outro:
            for sf in reversed(seg_features):
                if sf['orig_label'] == 'silence':
                    continue
                if sf['end_very_late']:
                    if sf['orig_label'] in ('verse', 'intro'):
                        _log.info("[SongFormer REMAP] outro: [%ds-%ds] %s -> outro "
                                  "(e=%.2f, pos=%.2f)" % (
                            sf['start'], sf['end'], sf['orig_label'],
                            sf['rel_energy'], sf['position_ratio']))
                        sf['orig_label'] = 'outro'
                        changed = True
                        break
                elif sf['end_late'] and sf['is_low_energy']:
                    if sf['orig_label'] in ('verse', 'intro'):
                        _log.info("[SongFormer REMAP] outro: [%ds-%ds] %s -> outro (低能量尾部)"
                                  % (sf['start'], sf['end'], sf['orig_label']))
                        sf['orig_label'] = 'outro'
                        changed = True
                        break

        # ── 6. 独立检测 bridge ──────────────────────────────────────────
        # 规则：在第 2-3 个 chorus 之后出现的中等能量 verse 段 → bridge
        # bridge 通常位于歌曲中部（15%-70%），持续时间 < 45s
        if not has_bridge and n_chorus >= 2:
            chorus_positions = [
                sf['position_ratio'] for sf in seg_features
                if sf['orig_label'] == 'chorus'
            ]
            if len(chorus_positions) >= 2:
                after_2nd_chorus = chorus_positions[1]  # 第二个 chorus 的位置
                for sf in seg_features:
                    if sf['position_ratio'] > after_2nd_chorus + 0.03 and                        sf['orig_label'] in ('verse', 'intro') and                        0.15 < sf['position_ratio'] < 0.70 and                        not sf['is_high_energy'] and not sf['is_low_energy'] and                        sf['duration'] < 45.0:
                        _log.info("[SongFormer REMAP] bridge: [%ds-%ds] %s -> bridge "
                                  "(e=%.2f, pos=%.2f, dur=%.1fs)" % (
                            sf['start'], sf['end'], sf['orig_label'],
                            sf['rel_energy'], sf['position_ratio'], sf['duration']))
                        sf['orig_label'] = 'bridge'
                        changed = True
                        break

        # ── 7. 独立检测 inst ─────────────────────────────────────────────
        # 规则：能量极低（低于 p25）且持续时间 >= 5s 的非 silence 段落 → inst
        if not has_inst:
            for sf in seg_features:
                if sf['orig_label'] in ('verse', 'intro', 'pre-chorus') and                    sf['is_low_energy'] and sf['duration'] >= 5.0:
                    _log.info("[SongFormer REMAP] inst: [%ds-%ds] %s -> inst "
                              "(e=%.2f, dur=%.1fs)" % (
                        sf['start'], sf['end'], sf['orig_label'],
                        sf['rel_energy'], sf['duration']))
                    sf['orig_label'] = 'inst'
                    changed = True
                    break

        # ── 8. 输出结果 ─────────────────────────────────────────────────
        remapped = []
        for sf in seg_features:
            remapped.append((sf['start'], sf['orig_label']))
        remapped.append((msa_list[-1][0], 'end'))

        _log.info("[SongFormer REMAP] 完成%s" % ("（无变化）" if not changed else ""))
        return remapped

    def analyze(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        分析音频文件（文件路径接口，向后兼容）

        Args:
            audio_path: 音频文件路径
        """
        self._ensure_models()
        os.chdir(SONGFORMER_ROOT)
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype='float32')
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return self._analyze_core(audio, sr)
        finally:
            if self._original_cwd:
                os.chdir(self._original_cwd)

    def analyze_array(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        分析音频数组（零拷贝接口，避免文件I/O）

        B-1 修复：processor.py 已有 y_song (numpy array)，
        直接传入即可跳过 librosa.load 的第三次加载。

        Args:
            audio: 音频数据 (numpy array, float32, mono)
            sample_rate: 采样率（必须 == INPUT_SAMPLING_RATE=24000）
        """
        self._ensure_models()
        os.chdir(SONGFORMER_ROOT)
        try:
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return self._analyze_core(audio, sample_rate)
        finally:
            if self._original_cwd:
                os.chdir(self._original_cwd)

    def _analyze_core(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        推理 + 后处理 + 标签映射的公共核心逻辑
        """
        log = logger.info if self.verbose else lambda *a, **k: None
        segments = self._run_inference_array(audio, time.time(), log)

        # 标签 → 中文
        mapped_segments = []
        for seg in segments:
            original = seg.get('label', 'verse')
            cn_label = SONGFORMER_LABEL_CN.get(original, original)
            mapped_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'label': cn_label,
                'confidence': seg.get('confidence', 0.8),
                'original_label': original,
            })

        if self.verbose:
            logger.info(f"[SongFormer] 分析完成: {len(mapped_segments)} 段")
            for s in mapped_segments:
                dur = s['end'] - s['start']
                logger.info(
                    f"  [{s['start']:>7.1f}s - {s['end']:>7.1f}s] "
                    f"({dur:>5.1f}s) {s['label']} ({s['original_label']})"
                )

        return mapped_segments

    def _run_inference(self, audio_path: str) -> List[Dict[str, Any]]:
        """SongFormer 推理主逻辑（从文件加载）"""
        log = logger.info if self.verbose else lambda *a, **k: None
        t0_total = time.time()

        # ── 0. 加载音频 ────────────────────────────────────────────────────
        log(f"[SongFormer] 加载音频: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=INPUT_SAMPLING_RATE, mono=True)

        return self._run_inference_array(audio, t0_total, log)

    def _run_inference_array(self, audio: np.ndarray, t0_total: float,
                             log=None) -> List[Dict[str, Any]]:
        """
        SongFormer 推理核心逻辑（从numpy数组输入）
        
        B-1 修复：跳过 librosa.load，直接用传入的 numpy array。
        processor.py 的 _analyze_song_segments_songformer 已有 y_resampled，
        无需再写临时文件再读回来。
        """
        if log is None:
            log = logger.info if self.verbose else lambda *a, **k: None

        # ── 1. 转 tensor ────────────────────────────────────────────────────
        total_samples = len(audio)
        total_duration = total_samples / INPUT_SAMPLING_RATE
        
        # 关键修复：强制深拷贝（librosa.resample 输出的数组可能带有特殊内存标志，
        # 导致 torch.from_numpy() 共享内存后 CUDA 操作产生全零 boundary_logits）
        # 即使 flags['C_CONTIGUOUS']=True 也可能有问题，必须 .copy()
        audio = np.array(audio, dtype=np.float32, copy=True)
        
        # DIAG: 音频数据统计
        log(f"[SongFormer DIAG-AUDIO] samples={total_samples}, dtype={audio.dtype}, "
            f"min={audio.min():.6f}, max={audio.max():.6f}, mean={audio.mean():.6f}, std={audio.std():.6f}")
        
        log(f"[SongFormer] 音频(数组): {total_duration:.1f}s, 采样率={INPUT_SAMPLING_RATE}")

        audio_tensor = torch.tensor(audio.copy(), dtype=torch.float32)  # 不用 from_numpy（避免 FP16 autocast NaN）
        # 预分配设备张量
        audio_dev = audio_tensor.to(self.device)
        del audio_tensor  # 释放CPU内存

        # ── 2. 推理循环（no_grad 禁用梯度计算）──────────────────────────────
        # 注意：数组路径使用纯 FP32（不用 autocast），因为 torch.from_numpy()
        # 创建的共享内存张量与 autocast 存在兼容性问题，会导致模型输出 NaN
        with torch.no_grad():
            total_frames = int(math.ceil(
                ((total_samples // INPUT_SAMPLING_RATE) // self.window_sec) * self.window_sec
                + self.window_sec
            ) * AFTER_DOWNSAMPLING_FRAME_RATES)

            logits_accum = {
                "function_logits": np.zeros([total_frames, NUM_CLASSES], dtype=np.float64),
                "boundary_logits": np.zeros([total_frames], dtype=np.float64),
            }
            logits_count = {
                "function_logits": np.zeros([total_frames, NUM_CLASSES], dtype=np.float64),
                "boundary_logits": np.zeros([total_frames], dtype=np.float64),
            }

            log(f"[SongFormer] 开始推理 (window={self.window_sec}s, hop={self.hop_sec}s)...")
            t0 = time.time()
            i = 0
            lens = 0
            frame_rate = AFTER_DOWNSAMPLING_FRAME_RATES
            
            # DIAG: 检查 GPU 上音频张量的实际值
            _diag_audio_dev = audio_dev[:1024] if audio_dev.shape[0] >= 1024 else audio_dev
            log(f"[SongFormer DIAG-TENSOR] audio_dev: "
                f"min={_diag_audio_dev.min():.6f}, max={_diag_audio_dev.max():.6f}, "
                f"mean={_diag_audio_dev.mean():.6f}")
            
            # 关键诊断：如果 GPU 张量全零，说明 torch.from_numpy 有问题
            if _diag_audio_dev.abs().max() < 1e-6:
                log("[SongFormer DIAG-TENSOR] WARNING: GPU 音频张量接近全零!")
                # 强制重新创建张量
                log("[SongFormer DIAG-TENSOR] 尝试 torch.tensor() 替代 torch.from_numpy()...")
                audio_dev = torch.tensor(
                    audio.cpu().numpy() if hasattr(audio, 'cpu') else audio,
                    dtype=torch.float32, device=self.device
                )
                log(f"[SongFormer DIAG-TENSOR] 重试后: min={audio_dev.min():.6f}, max={audio_dev.max():.6f}")

            while True:
                start_idx = i * INPUT_SAMPLING_RATE
                end_idx = min((i + self.window_sec) * INPUT_SAMPLING_RATE, total_samples)

                if start_idx >= total_samples:
                    break
                if end_idx - start_idx <= 1024:
                    i += self.hop_sec
                    continue

                audio_seg = audio_dev[start_idx:end_idx]

                # 长窗嵌入
                # 重要：torch.from_numpy 创建的张量 + autocast FP16 会产生 NaN
                # librosa.load 返回的数组则正常。根因不明但已复现。
                # 解决方案：数组路径强制 FP32（避免 autocast NaN）
                try:
                    # 纯 FP32 模式（数组路径）
                    muq_out = self._muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                    muq_embd = muq_out["hidden_states"][10]
                    del muq_out

                    _, mf_states = self._musicfm.get_predictions(audio_seg.unsqueeze(0))
                    musicfm_embd = mf_states[10]
                    del mf_states
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        log(f"  CUDA OOM 错误: {str(e)}")
                        log(f"  当前窗口: {i}~{i + self.window_sec}秒")
                        log(f"  尝试清理缓存并重试...")
                        torch.cuda.empty_cache()
                        # 跳过当前窗口，继续下一个
                        continue
                    else:
                        raise
                # 短窗嵌入（30s 切分）
                wraped_muq_30s = []
                wraped_mf_30s = []

                for idx_30s in range(i, i + self.window_sec, 30):
                    s30 = idx_30s * INPUT_SAMPLING_RATE
                    e30 = min(
                        (idx_30s + 30) * INPUT_SAMPLING_RATE,
                        total_samples,
                    )
                    if s30 >= total_samples or e30 - s30 <= 1024:
                        continue

                    try:
                        # 纯 FP32（同长窗）
                        muq_30s_out = self._muq(
                            audio_dev[s30:e30].unsqueeze(0), output_hidden_states=True
                        )
                        wraped_muq_30s.append(muq_30s_out["hidden_states"][10])
                        del muq_30s_out

                        _, mf_30s_states = self._musicfm.get_predictions(
                            audio_dev[s30:e30].unsqueeze(0)
                        )
                        wraped_mf_30s.append(mf_30s_states[10])
                        del mf_30s_states

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            log(f"  CUDA OOM 错误: {str(e)}")
                            log(f"  当前窗口: {i}~{i + self.window_sec}秒")
                            log(f"  尝试清理缓存并重试...")
                            torch.cuda.empty_cache()
                            # 跳过当前窗口，继续下一个
                            continue
                        else:
                            raise
                wraped_muq_30s = (
                    torch.concatenate(wraped_muq_30s, dim=1) if wraped_muq_30s else muq_embd
                )
                wraped_mf_30s = (
                    torch.concatenate(wraped_mf_30s, dim=1) if wraped_mf_30s else musicfm_embd
                )

                all_embds = [
                    wraped_mf_30s,
                    wraped_muq_30s,
                    musicfm_embd,
                    muq_embd,
                ]

                # 对齐长度
                if len(all_embds) > 1:
                    embd_lens = [x.shape[1] for x in all_embds]
                    min_len = min(embd_lens)
                    if abs(max(embd_lens) - min_len) > 4:
                        log(f"  警告: 嵌入长度差异 {embd_lens}，取min={min_len}")
                    for idx in range(len(all_embds)):
                        all_embds[idx] = all_embds[idx][:, :min_len, :]

                embd = torch.concatenate(all_embds, axis=-1)
                del all_embds, wraped_muq_30s, wraped_mf_30s, muq_embd, musicfm_embd

                # 调用 model.infer
                dataset_ids_t = torch.Tensor(self._dataset_ids).to(
                    self.device, dtype=torch.long
                )
                label_mask_t = (
                    torch.from_numpy(self._label_mask)
                    .to(self.device, dtype=torch.bool)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                try:
                    # 直接调 forward（跳过单窗口后处理，累积 logits 后统一后处理）
                    embd_downsampled = self._model.mixed_win_downsample(embd)
                    embd_norm = self._model.input_norm(embd_downsampled)
                    feat = self._model.down_sample_conv(embd_norm)
                    dataset_prefix = self._model.dataset_class_prefix(dataset_ids_t)
                    dataset_prefix_expand = dataset_prefix.unsqueeze(1).expand(feat.size(0), 1, -1)
                    feat = self._model.AddFuse(x=feat, cond=dataset_prefix_expand)
                    feat = self._model.transformer(x=feat, src_key_padding_mask=None)
                    function_logits = self._model.function_head(feat)
                    boundary_logits = self._model.boundary_head(feat).squeeze(-1)

                    # label mask
                    if boundary_logits.dim() == 0:
                        boundary_logits = boundary_logits.unsqueeze(0)
                    # 修复: broadcast mask correctly over (1,T,128)
                    # expanded_mask: (1,1,128) → expand → (1,T,128)
                    # squeeze(0) → (T,128)，masked_fill broadcasting: (1,T,128)+(T,128) → OK
                    expanded_mask = label_mask_t.expand(-1, function_logits.size(1), -1)
                    function_logits = function_logits.masked_fill(expanded_mask.squeeze(0), float('-inf'))

                    func_np = function_logits.float().cpu().detach().numpy()
                    bound_np = boundary_logits.float().cpu().detach().numpy()
                    
                    # 关键诊断：检查模型原始输出（第1个窗口只打一次）
                    if i == 0:
                        log(f"[SongFormer DIAG-RAW] Window 0 raw output:")
                        log(f"  function_logits: shape={func_np.shape}, "
                            f"min={func_np.min():.4f}, max={func_np.max():.4f}")
                        log(f"  boundary_logits: shape={bound_np.shape}, "
                            f"min={bound_np.min():.4f}, max={bound_np.max():.4f}, "
                            f"mean={bound_np.mean():.4f}")
                        # 检查是否有非零值
                        _nz_bound = np.count_nonzero(bound_np)
                        log(f"  boundary_logits non-zero count: {_nz_bound}/{len(bound_np)}")
                        if _nz_bound == 0:
                            log("[SongFormer DIAG-RAW] !!! boundary_head 输出全零 !!!")
                            log(f"  embd shape: {embd.shape}, feat shape: {feat.shape}")
                    
                    if func_np.ndim == 3:
                        func_np = func_np[0]
                    if bound_np.ndim == 2:
                        bound_np = bound_np[0]

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        log(f"  CUDA OOM 错误: {str(e)}")
                        log(f"  当前窗口: {i}~{i + self.window_sec}秒")
                        log(f"  尝试清理缓存并重试...")
                        torch.cuda.empty_cache()
                        # 跳过当前窗口，继续下一个
                        continue
                    else:
                        raise
                del embd

                # 累积
                start_frame = int(i * frame_rate)
                hop_frames = int(self.hop_sec * frame_rate)
                end_frame = min(start_frame + hop_frames, total_frames)
                actual = min(end_frame - start_frame, func_np.shape[0], bound_np.shape[0])

                logits_accum["function_logits"][start_frame:start_frame+actual] += func_np[:actual]
                logits_accum["boundary_logits"][start_frame:start_frame+actual] += bound_np[:actual]
                logits_count["function_logits"][start_frame:start_frame+actual] += 1
                logits_count["boundary_logits"][start_frame:start_frame+actual] += 1
                lens = max(lens, start_frame + actual)

                if self.verbose and (i // self.hop_sec) % 5 == 0:
                    log(f"  窗口 {i // self.hop_sec + 1}: {time.time() - t0:.1f}秒 (累计)")

                i += self.hop_sec

        # 平均
        valid_f = logits_count["function_logits"] > 0
        valid_b = logits_count["boundary_logits"] > 0
        logits_accum["function_logits"][valid_f] /= logits_count["function_logits"][valid_f]
        logits_accum["boundary_logits"][valid_b] /= logits_count["boundary_logits"][valid_b]

        # 截断
        func_final = logits_accum["function_logits"][:lens].astype(np.float32)
        bound_final = logits_accum["boundary_logits"][:lens].astype(np.float32)
        duration = lens / frame_rate

        log(f"[SongFormer] 推理完成 ({time.time()-t0:.1f}秒)")
        log(f"[SongFormer] logits shape: function={func_final.shape}, boundary={bound_final.shape}")

        # ── 2.5 原始标签概率诊断（后处理前）──────────────────────────
        LABEL_NAMES_8CLASS = [
            'silence', 'intro', 'verse', 'chorus',
            'bridge', 'inst', 'prechorus', 'outro',
        ]
        exp_l = np.exp(func_final - np.max(func_final, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        # 每60秒窗口的 top-3 标签
        win_sec = 60; n_win = int(duration // win_sec) + 1
        log(f"[SongFormer DIAG] Raw label probabilities per {win_sec}s window:")
        for w in range(n_win):
            ws, we = int(w * win_sec * frame_rate), min(int((w+1)*win_sec*frame_rate), len(probs))
            if we <= ws: break
            avg_p = np.mean(probs[ws:we], axis=0)
            top3 = np.argsort(avg_p)[::-1][:3]
            names = ', '.join([f"{LABEL_NAMES_8CLASS[i]}:{avg_p[i]:.2f}" for i in top3 if i < len(LABEL_NAMES_8CLASS)])
            log(f"  [{w*win_sec/60:.0f}m{(w*win_sec)%60:02.0f}s-{(w+1)*win_sec/60:.0f}m{((w+1)*win_sec)%60:02.0f}s] {names}")

        # ── 2.6 DIAG: 后处理前检查 boundary_logits ───────────────────────
        log(f"[SongFormer DIAG-POST] bound_final: shape={bound_final.shape}, "
            f"min={np.nanmin(bound_final):.4f}, max={np.nanmax(bound_final):.4f}, "
            f"mean={np.nanmean(bound_final):.4f}")
        _sig_bound = 1 / (1 + np.exp(-bound_final))  # sigmoid
        log(f"[SongFormer DIAG-POST] sigmoid(boundary): min={_sig_bound.min():.4f}, "
            f"max={_sig_bound.max():.4f}, mean={_sig_bound.mean():.4f}")

        # ── 3. 后处理 ──────────────────────────────────────────────────────
        from postprocessing.functional import postprocess_functional_structure

        logits_dict = {
            "function_logits": torch.from_numpy(func_final).unsqueeze(0),
            "boundary_logits": torch.from_numpy(bound_final).unsqueeze(0),
        }

        msa_output = postprocess_functional_structure(
            logits=logits_dict, config=self._hp
        )
        log(f"[SongFormer] 原始分段: {len(msa_output)-1} 段")

        msa_output = self._rule_post_processing(msa_output)
        log(f"[SongFormer] 规则后处理后: {len(msa_output)-1} 段")

        # ── 2.7 声学特征二次映射（修复 Live 演唱会 domain gap）──────
        # 关键：必须在 _merge_adjacent_same_label 之前执行！
        # 因为 _merge_adjacent_same_label 会把 intro 和 verse 合并，
        # 导致 remap 无法保留 intro。调整顺序后：
        # 原始输出 → remap（识别intro/outro/bridge） → 合并 → 过短合并 → 长段切分
        msa_output = self._remap_live_labels(
            msa_output, func_final, bound_final, frame_rate, duration
        )
        log(f"[SongFormer] 声学重映射后: {len(msa_output)-1} 段")

        # 合并相邻同标签
        msa_output = self._merge_adjacent_same_label(msa_output)
        msa_output = self._merge_short_segments(msa_output, min_duration=8.0)
        msa_output = self._split_long_segments(msa_output, max_duration=90.0)
        log(f"[SongFormer] 合并后: {len(msa_output)-1} 段")

        # 计算边界置信度（np 已在模块顶部导入）
        boundary_probs = 1 / (1 + np.exp(-bound_final))  # sigmoid
        func_probs = np.exp(func_final) / np.sum(np.exp(func_final), axis=-1, keepdims=True)  # softmax
        
        # 转 dict，添加置信度过滤
        segments = []
        for idx in range(len(msa_output) - 1):
            start_frame = int(msa_output[idx][0] * frame_rate)
            end_frame = int(msa_output[idx + 1][0] * frame_rate)
            if start_frame >= len(boundary_probs):
                start_frame = max(0, len(boundary_probs) - 1)
            if end_frame > len(boundary_probs):
                end_frame = len(boundary_probs)
            
            # 计算段落内平均置信度
            frame_confidence = boundary_probs[start_frame:end_frame]
            func_confidence = func_probs[start_frame:end_frame].max(axis=-1)
            if len(frame_confidence) > 0:
                avg_bound_conf = np.mean(frame_confidence)
                avg_func_conf = np.mean(func_confidence)
                confidence = avg_bound_conf * avg_func_conf
            else:
                confidence = 0.5
            
            label = msa_output[idx][1]
            # 置信度低于阈值时降级，但 chorus/bridge/outro/prechorus/intro 等结构标签保留
            # 只有 verse/inst 才降级为 verse（因为这两者对下游剪辑影响较小）
            if confidence < 0.2 and label in ('verse', 'inst'):
                label = "verse"
            # nan 置信度的 inst 也降级
            elif (np.isnan(confidence) or confidence < 0.01) and label == 'inst':
                label = "verse"
            # intro 即使低置信度也保留（结构重要）


            segments.append({
                "start": round(msa_output[idx][0], 2),
                "end": round(msa_output[idx + 1][0], 2),
                "label": label,
                "confidence": round(confidence, 3),
            })

        log(f"[SongFormer] 总耗时: {time.time()-t0_total:.1f}秒")

        return segments

    # ── 便捷方法 ──────────────────────────────────────────────────────────────

    def get_song_boundaries(self, audio_path: str) -> List[float]:
        """
        只获取歌曲边界时间点（不获取标签）
        用于替换 SongBoundaryDetector
        """
        segments = self.analyze(audio_path)
        boundaries = [s['end'] for s in segments]
        return sorted(set(boundaries))

    def get_chorus_segments(
        self,
        audio_path: str,
        min_duration: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """
        只获取副歌段落（用于辅助判断）

        Args:
            min_duration: 最小副歌时长（秒）
        """
        segments = self.analyze(audio_path)
        chorus_segs = [
            s for s in segments
            if s['label'] == 'chorus' and (s['end'] - s['start']) >= min_duration
        ]
        return chorus_segs
