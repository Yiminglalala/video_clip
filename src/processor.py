"""
Live演唱会视频智能切片工具 - 主处理器 v3.0

新架构：
1. 先检测视频边界（OCR标题画面）作为主要歌曲边界
2. 再用音频特征作为辅助验证
3. 合并后确定歌曲
4. 每首歌单独分析段落并分类
5. 利用歌曲内位置上下文辅助分类
"""

import os
import sys
import shutil
import time
import tempfile
import uuid
import logging
import math
import re
import gc
import threading
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np

from src.audio_analyzer import (
    AudioAnalyzer, Segment, SongInfo, AnalysisResult,
    LABEL_SILENCE, LABEL_SPEECH, LABEL_TALK, LABEL_CROWD,
    LABEL_AUDIENCE, LABEL_INTRO, LABEL_VERSE, LABEL_CHORUS,
    LABEL_OUTRO, LABEL_INTERLUDE, LABEL_SOLO, LABEL_OTHER, LABEL_CN,
)
from src.ffmpeg_processor import FFmpegProcessor
from src.doubao_api import DoubaoASR, format_result
from src.output_spec import (
    DEFAULT_LANDSCAPE_RESOLUTION,
    OutputResolutionSpec,
    STANDARD_VIDEO_OUTPUT_ARGS,
    build_ass_filter_value,
    build_cover_crop_filter,
    normalize_landscape_resolution_choice,
    resolve_output_resolution_spec,
)

logger = logging.getLogger(__name__)

_GPU_HEAVY_TASK_LOCK = threading.Lock()


@dataclass
class ProcessingConfig:
    """处理配置"""
    output_dir: str = r"D:\video_clip\output"
    min_segment_duration: float = 8.0  # 最小片段时长(秒)
    max_segment_duration: float = 15.0   # 最大片段时长(秒)
    enhance_audio: bool = True
    enable_vad: bool = True
    enable_demucs: bool = True
    demucs_model: str = "htdemucs_ft"  # P0升级：默认用最高质量模型
    enable_songformer: bool = True      # P1新增：SongFormer SOTA 段落分析（GPU加速）
    songformer_device: str = "auto"     # P1: cuda / cpu / auto
    songformer_window: int = 60         # P1: 推理窗口大小（秒）——显存8GB下最大60s
    songformer_hop: int = 30           # P1: 跳跃步长（30s，50%重叠）
    strict_songformer: bool = True      # SongFormer 失败时直接报错，不降级手工分类
    concert: Optional[str] = None
    cut_mode: str = "fast"
    audio_sample_rate: int = 22050
    keep_intermediate: bool = False
    # ── 字幕配置 ──
    enable_subtitle: bool = True       # 烧录 ASR 字幕
    subtitle_model: str = "small"       # Whisper 模型: tiny / base / small
    subtitle_mode: str = "word"         # "word"=逐词 / "sentence"=整句
    subtitle_orientation: str = "auto"  # "auto" / "portrait" / "landscape"
    lyrics_hint_file: Optional[str] = None  # 可选：Song_01 -> 歌名/歌手 映射 JSON
    min_duration_limit: float = 8.0
    max_duration_limit: float = 15.0
    export_fps: int = 60
    landscape_resolution_choice: str = DEFAULT_LANDSCAPE_RESOLUTION
    source_orientation: str = "auto"


class LiveVideoProcessor:
    """
    演唱会视频处理器 v3.0
    
    新流程：
    1. 视频边界检测（OCR标题画面）
    2. 音频边界检测（能量谷值）
    3. 合并边界确定歌曲
    4. 每首歌单独分析
    5. 歌曲内位置上下文辅助分类
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.audio_analyzer = AudioAnalyzer(
            sample_rate=self.config.audio_sample_rate,
            enable_vad=self.config.enable_vad,
            enable_demucs=self.config.enable_demucs,
        )
        self.ffmpeg_processor = FFmpegProcessor()
        self._temp_files = []
        # P1: SongFormer 分析器（懒加载，不预加载）
        self._songformer_analyzer = None
        self._ocr_title_frames: List[dict] = []
        self._boundary_scores: Dict[float, float] = {}
        self._last_export_metadata: List[Dict[str, Any]] = []
        self._disable_song_identify: bool = True
        self._disable_lyrics_identify: bool = True
        self._cached_asr_results: Dict[int, dict] = {}  # {song_index: doubao_asr_result}
        self._gpu_task_lock = _GPU_HEAVY_TASK_LOCK

    def _try_get_cuda_memory_stats(self) -> Optional[Dict[str, float]]:
        """获取当前 CUDA 显存状态（MB），不可用时返回 None。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            allocated = float(torch.cuda.memory_allocated() / (1024 ** 2))
            reserved = float(torch.cuda.memory_reserved() / (1024 ** 2))
            free, total = torch.cuda.mem_get_info()
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "free_mb": float(free / (1024 ** 2)),
                "total_mb": float(total / (1024 ** 2)),
            }
        except Exception:
            return None

    def _log_gpu_memory(self, stage: str) -> None:
        stats = self._try_get_cuda_memory_stats()
        if not stats:
            logger.info(f"[GPU] {stage}: CUDA unavailable")
            return
        logger.info(
            "[GPU] %s: allocated=%.1fMB reserved=%.1fMB free=%.1fMB total=%.1fMB",
            stage,
            stats["allocated_mb"],
            stats["reserved_mb"],
            stats["free_mb"],
            stats["total_mb"],
        )

    def _cleanup_gpu_stage(self, stage: str, unload_songformer: bool = False) -> None:
        """标准 GPU 清理序列：del 引用(按调用方执行) + gc + sync + empty_cache。"""
        if unload_songformer:
            try:
                from src.songformer_analyzer import SongFormerAnalyzer
                SongFormerAnalyzer.reset_instance()
            except Exception as e:
                logger.warning("[GPU] unload songformer failed at %s: %s", stage, e)
            self._songformer_analyzer = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning("[GPU] cleanup warning at %s: %s", stage, e)

        self._log_gpu_memory(f"{stage} (post-clean)")

    @staticmethod
    def _detect_orientation_from_video_info(video_info: Optional[Dict[str, Any]]) -> str:
        width = int((video_info or {}).get("width") or 0)
        height = int((video_info or {}).get("height") or 0)
        if width > 0 and height > 0:
            return "portrait" if height >= width else "landscape"
        return "landscape"

    def _resolve_output_spec(
        self,
        video_path: Optional[str] = None,
        video_info: Optional[Dict[str, Any]] = None,
    ) -> OutputResolutionSpec:
        orientation = str(getattr(self.config, "source_orientation", "auto") or "auto").strip().lower()
        if orientation not in {"portrait", "landscape"}:
            if video_info is None and video_path:
                try:
                    video_info = self.ffmpeg_processor._get_video_info(video_path)
                except Exception:
                    video_info = None
            orientation = self._detect_orientation_from_video_info(video_info)
        landscape_choice = normalize_landscape_resolution_choice(
            getattr(self.config, "landscape_resolution_choice", DEFAULT_LANDSCAPE_RESOLUTION)
        )
        return resolve_output_resolution_spec(orientation, landscape_choice)

    @property
    def songformer_analyzer(self):
        """懒加载 SongFormer"""
        if self._songformer_analyzer is None:
            if self.config.enable_songformer:
                try:
                    import torch
                    cuda_available = torch.cuda.is_available()
                except Exception:
                    cuda_available = False
                
                # 根据 CUDA 可用性调整窗口大小
                if not cuda_available or self.config.songformer_device == "cpu":
                    window_sec = 30  # CPU 模式下使用更小的窗口
                    hop_sec = 15
                    logger.info("CUDA 不可用，使用 CPU 模式优化配置 (window=30s, hop=15s)")
                else:
                    window_sec = self.config.songformer_window
                    hop_sec = self.config.songformer_hop
                
                from src.songformer_analyzer import SongFormerAnalyzer
                self._songformer_analyzer = SongFormerAnalyzer.get_instance(
                    device=self.config.songformer_device,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                    verbose=True,
                )
                logger.info(
                    f"SongFormer 已加载 (device={getattr(self._songformer_analyzer, 'device', self.config.songformer_device)}, "
                    f"window={window_sec}s, hop={hop_sec}s)"
                )
        return self._songformer_analyzer
    
    def _add_temp_file(self, path: str):
        """添加临时文件到清理列表"""
        if path and os.path.exists(path):
            self._temp_files.append(path)
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    logger.debug(f"已清理临时文件: {f}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {f} - {e}")
        self._temp_files = []
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """使用ffprobe获取音频时长，避免librosa加载问题"""
        import subprocess
        import json
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            if result.returncode != 0 or not result.stdout:
                logger.warning("无法获取音频时长，默认返回60秒")
                return 60.0
            info = json.loads(result.stdout)
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    return float(stream.get('duration', 0.0))
            
            logger.warning("无法获取音频时长，默认返回60秒")
            return 60.0
        except Exception as e:
            logger.warning(f"获取音频时长失败: {e}，默认返回60秒")
            return 60.0

    def _load_song_hint(self, song_name: str, singer: str = None) -> Dict[str, str]:
        """读取可选歌单提示，解决短现场片段 Shazam 不命中的情况。"""
        import json

        candidates = []
        if self.config.lyrics_hint_file:
            candidates.append(Path(self.config.lyrics_hint_file))
        candidates.extend([
            Path(self.config.output_dir) / "lyrics_hints.json",
            Path(__file__).resolve().parents[1] / "lyrics_hints.json",
        ])

        for hint_file in candidates:
            if not hint_file.exists():
                continue
            try:
                data = json.loads(hint_file.read_text(encoding="utf-8-sig"))
            except Exception as e:
                logger.warning(f"读取歌词提示失败: {hint_file} - {e}")
                continue

            entry = data.get(song_name)
            if isinstance(entry, str):
                return {"title": entry, "artist": singer or ""}
            if isinstance(entry, dict):
                return {
                    "title": str(entry.get("title") or ""),
                    "artist": str(entry.get("artist") or singer or ""),
                    "lyrics_text": str(entry.get("lyrics_text") or ""),
                    "start_line": int(entry.get("start_line") or 0),
                    "end_line": entry.get("end_line"),
                    "auto_locate": bool(entry.get("auto_locate", True)),
                    "clip_start_sec": entry.get("clip_start_sec"),
                    "offset_sec": float(entry.get("offset_sec") or 0.0),
                }

        return {"artist": singer or "", "title": ""}
    
    def analyze_video(
        self,
        video_path: str,
        singer: Optional[str] = None,
        concert: Optional[str] = None,
        progress_callback=None
    ) -> Tuple[AnalysisResult, List[Dict[str, Any]]]:
        """
        分析视频（只做结构分析，不导出视频文件）

        Args:
            video_path: 视频文件路径
            singer: 歌手名称（如果为None则从文件名解析）
            concert: 演唱会名称
            progress_callback: 进度回调函数

        Returns:
            (分析结果, 最终导出片段列表)
        """
        logger.info(f"开始分析视频: {video_path}")
        if not self._gpu_task_lock.acquire(blocking=False):
            raise RuntimeError("已有 GPU 重任务在运行，请等待当前任务完成后再启动新的处理任务。")
        self._log_gpu_memory("analyze-start")

        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False

        if self.config.enable_songformer and not cuda_ok:
            logger.warning("[GPU] CUDA unavailable: SongFormer 将使用 CPU，速度会明显下降。")
        if self.config.enable_demucs and not cuda_ok:
            logger.warning("[GPU] CUDA unavailable: Demucs 将使用 CPU，速度会明显下降。")

        # 1. 确定歌手和演唱会名称
        if singer is None:
            singer, auto_concert = self.audio_analyzer.parse_singer_and_concert(os.path.basename(video_path))
        else:
            auto_concert = None

        if concert is None and auto_concert:
            concert = auto_concert
            logger.info(f"自动识别演唱会: {concert}")

        logger.info(f"识别歌手: {singer}")
        if concert:
            logger.info(f"识别演唱会: {concert}")

        video_path_work = video_path
        temp_video_path = None
        audio_path = None

        try:
            unicode_path = any(ord(ch) > 127 for ch in str(video_path))
            if progress_callback:
                progress_callback(0.01, "正在检查输入视频...")

            # 3. 提取音频（先直接用原路径，失败后再回退复制）
            if progress_callback:
                progress_callback(0.05, "正在提取音频...")

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            self._add_temp_file(audio_path)

            extract_result = self.ffmpeg_processor.extract_audio(
                video_path_work,
                audio_path,
                sample_rate=self.config.audio_sample_rate
            )
            if not extract_result.success and unicode_path and temp_video_path is None:
                logger.warning("extract audio failed on unicode path, fallback to temp copy")
                if progress_callback:
                    progress_callback(0.04, "检测到路径兼容问题，正在准备临时副本...")
                temp_dir = tempfile.gettempdir()
                video_ext = os.path.splitext(video_path)[1]
                temp_video_path = os.path.join(temp_dir, f"temp_video_{os.getpid()}_{int(time.time())}{video_ext}")
                shutil.copy2(video_path, temp_video_path)
                video_path_work = temp_video_path
                self._add_temp_file(temp_video_path)
                if progress_callback:
                    progress_callback(0.05, "正在提取音频（临时副本）...")
                extract_result = self.ffmpeg_processor.extract_audio(
                    video_path_work,
                    audio_path,
                    sample_rate=self.config.audio_sample_rate
                )

            if not extract_result.success:
                raise Exception(f"音频提取失败: {extract_result.error_message}")

            logger.info(f"音频提取完成: {audio_path}")

            # 全局转换一次AED用的WAV文件，避免每首歌都转换
            aed_wav_path = None
            if self.audio_analyzer is not None and self.audio_analyzer.vad is not None:
                try:
                    import subprocess

                    aed_wav_path = tempfile.mktemp(suffix=".wav")
                    logger.info(f"[AED] 全局转换 {os.path.basename(audio_path)} -> WAV")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", audio_path,
                        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", aed_wav_path
                    ], capture_output=True, check=True)
                    self._add_temp_file(aed_wav_path)
                except Exception as aed_convert_err:
                    logger.warning(f"[AED] 全局转换失败: {aed_convert_err}")
                    aed_wav_path = None

            # ═══════════════════════════════════════════════════════════════
            # 检测歌曲边界
            # ═══════════════════════════════════════════════════════════════

            if progress_callback:
                progress_callback(0.10, "检测视频边界（OCR标题画面）...")

            video_boundaries = self._detect_video_boundaries(video_path_work, progress_callback)
            logger.info(f"视频边界（标题画面）: {len(video_boundaries)}个 -> {video_boundaries}")

            if progress_callback:
                progress_callback(0.15, "检测音频边界（能量谷值）...")
            audio_boundaries = self._detect_audio_boundaries(audio_path, progress_callback)
            logger.info(f"音频边界（能量谷值）: {len(audio_boundaries)}个 -> {audio_boundaries}")

            if progress_callback:
                progress_callback(0.22, "MSAF音乐结构分析...")
            msaf_boundaries = self._detect_msaf_boundaries(audio_path, progress_callback)
            logger.info(f"MSAF 边界（结构分析）: {len(msaf_boundaries)}个 -> {msaf_boundaries}")

            if progress_callback:
                progress_callback(0.30, "合并边界确定歌曲...")

            song_boundaries = self._merge_boundaries(
                video_boundaries,
                audio_boundaries,
                msaf_boundaries,
                audio_path
            )
            self._log_gpu_memory("stage-A-boundary")
            self._cleanup_gpu_stage("stage-A-boundary")
            logger.info(f"合并后歌曲边界: {len(song_boundaries)}首 -> {song_boundaries}")

            # 创建歌曲结构（SongFormer 分析）
            logger.info("准备调用 _create_songs...")
            self._cached_asr_results = {}  # 清除ASR缓存
            songs = self._create_songs(
                song_boundaries,
                audio_path,
                progress_callback,
                singer=singer,
                aed_wav_path=aed_wav_path,
            )
            logger.info(f"_create_songs 完成，创建了 {len(songs)} 首歌曲")

            # 创建分析结果
            result = AnalysisResult(
                songs=songs,
                singer=singer,
                total_duration=songs[-1].end_time if songs else 0,
                audio_info={},
                analysis_time=0.0,
            )

            # 构建最终导出片段（合并/拆分/过滤后）
            if progress_callback:
                progress_callback(0.85, "构建最终导出片段...")

            min_duration = float(getattr(self.config, "min_duration_limit", 8.0) or 8.0)
            max_duration = float(getattr(self.config, "max_duration_limit", 15.0) or 15.0)
            min_duration = max(8.0, min(20.0, min_duration))
            max_duration = max(8.0, min(20.0, max_duration))
            if max_duration < min_duration:
                max_duration = min_duration

            export_segments: List[Dict[str, Any]] = []
            for song in result.songs:
                export_segments.extend(self._build_export_segments_for_song(song, min_duration, max_duration))

            # 烧录字幕到完整视频（如果启用字幕）
            subtitled_video_path = None
            if self.config.enable_subtitle and self._cached_asr_results:
                if progress_callback:
                    progress_callback(0.95, "正在烧录字幕到完整视频...")
                try:
                    video_name = os.path.basename(video_path)
                    video_name_no_ext = os.path.splitext(video_name)[0]
                    output_dir = os.path.join(self.config.output_dir, f"subtitled_{datetime.now().strftime('%Y%m%d_%H%M')}")
                    subtitled_video_path = os.path.join(output_dir, f"{video_name_no_ext}_subtitled.mp4")
                    os.makedirs(output_dir, exist_ok=True)
                    subtitled_video_path = self._burn_subtitles_to_full_video(
                        video_path, subtitled_video_path, result.songs
                    )
                except Exception as e:
                    logger.warning(f"字幕烧录失败: {e}")
                    subtitled_video_path = None

            if progress_callback:
                progress_callback(1.0, "分析完成!")

            logger.info(f"分析完成，共 {len(result.songs)} 首歌，{len(export_segments)} 个最终片段")
            return result, export_segments, subtitled_video_path

        except Exception as e:
            import traceback
            logger.error(f"分析失败: {e}\n{traceback.format_exc()}")
            raise

        finally:
            self._cleanup_gpu_stage("analyze-final")
            try:
                self._gpu_task_lock.release()
            except RuntimeError:
                pass
            if not self.config.keep_intermediate:
                self._cleanup_temp_files()

    def export_video_segments(
        self,
        video_path: str,
        export_segments: List[Dict[str, Any]],
        progress_callback=None,
        singer: str = None
    ) -> List[str]:
        """
        导出视频切片（ffmpeg 切片，实际写文件）

        Args:
            video_path: 原始视频文件路径
            export_segments: 最终导出片段列表（来自 analyze_video）
            progress_callback: 进度回调函数
            singer: 歌手名称

        Returns:
            输出文件路径列表
        """
        output_files: List[str] = []
        self._last_export_metadata = []

        version_folder = datetime.now().strftime("%Y%m%d_%H%M")
        concert = self.config.concert or ""
        singer_safe = self._sanitize_filename_component(singer or "output", "output")
        concert_safe = self._sanitize_filename_component(concert or "", "")
        if concert_safe:
            top_folder = f"{version_folder}_{singer_safe}_{concert_safe}"
        else:
            top_folder = f"{version_folder}_{singer_safe}"

        cached_video_info = self.ffmpeg_processor._get_video_info(video_path)
        output_spec = self._resolve_output_spec(video_path, cached_video_info)
        total_segments = len(export_segments)
        processed = 0
        date_prefix = datetime.now().strftime("%Y%m%d")
        global_seq = 0

        # 字幕相关（不再预加载全量音频，改用ASR缓存）
        for item in export_segments:
            song: SongInfo = item["song"]
            seg: Segment = item["segment"]
            export_type = item["type"]
            is_highlight = bool(item["is_highlight"])
            start_time = float(item["start"])
            end_time = float(item["end"])

            setattr(seg, "export_type", export_type)
            setattr(seg, "is_highlight", is_highlight)

            raw_song_title = getattr(song, 'song_title', '') or "未知歌曲"
            song_folder = self._sanitize_filename_component(raw_song_title, "未知歌曲")
            song_dir = Path(self.config.output_dir) / top_folder / song_folder
            song_dir.mkdir(parents=True, exist_ok=True)

            song_title_for_name = self._sanitize_filename_component(raw_song_title, "未知歌曲")
            global_seq += 1
            seq = global_seq

            filename = f"{date_prefix}_{song_title_for_name}_{seq:03d}_{export_type}.mp4"
            output_path = song_dir / filename

            cut_result = self.ffmpeg_processor.cut_video(
                video_path,
                start_time,
                end_time,
                str(output_path),
                mode=self.config.cut_mode,
                video_info=cached_video_info,
                output_spec=output_spec,
            )
            if not cut_result.success:
                logger.warning(f"clip failed, skipped: {output_path} - {cut_result.error_message}")
                continue

            if self.config.enable_subtitle:
                # 从ASR缓存取该歌的识别结果，避免重复调API
                song_idx = song.song_index
                cached_asr = self._cached_asr_results.get(song_idx)
                if cached_asr and not cached_asr.get("error"):
                    subtitle_tmp = output_path.with_name(f"{output_path.stem}_subtitled{output_path.suffix}")
                    ok, res = self._generate_subtitles_from_cached_asr(
                        video_path=str(output_path),
                        asr_result=cached_asr,
                        seg_start=start_time - song.start_time,
                        seg_end=end_time - song.start_time,
                        output_path=str(subtitle_tmp),
                        output_spec=output_spec,
                    )
                    if not ok:
                        logger.warning(f"[Doubao] subtitle failed for {output_path.name}: {res}")
                    else:
                        try:
                            shutil.move(str(subtitle_tmp), str(output_path))
                        except Exception as move_err:
                            logger.warning(f"[subtitle] finalize move failed: {move_err}")
                            output_path = subtitle_tmp
                else:
                    logger.debug(f"[subtitle] no cached ASR for song {song_idx}, skip subtitle")

            output_files.append(str(output_path))
            processed += 1

            export_meta = {
                "path": str(output_path),
                "type": export_type,
                "songformer_label": item.get("songformer_label", ""),
                "highlight": is_highlight,
                "song_title": raw_song_title,
                "song_name": song.song_name,
                "start": start_time,
                "end": end_time,
            }
            self._last_export_metadata.append(export_meta)
            logger.info(
                f"[export] {output_path.name} | song={raw_song_title} | type={export_type} "
                f"| highlight={is_highlight} | {start_time:.2f}s-{end_time:.2f}s"
            )

            if progress_callback:
                progress = processed / max(1, total_segments)
                progress_callback(progress, f"导出切片: {raw_song_title} - {export_type}")

        return output_files

    def process_video(
        self,
        video_path: str,
        singer: Optional[str] = None,
        concert: Optional[str] = None,
        progress_callback=None
    ) -> Tuple[AnalysisResult, List[str]]:
        """
        处理完整视频（分析+导出，向后兼容）

        Args:
            video_path: 视频文件路径
            singer: 歌手名称
            concert: 演唱会名称
            progress_callback: 进度回调函数

        Returns:
            (分析结果, 输出文件列表)
        """
        result, export_segments = self.analyze_video(
            video_path, singer=singer, concert=concert, progress_callback=progress_callback
        )
        output_files = self.export_video_segments(
            video_path, export_segments, progress_callback=progress_callback, singer=singer
        )
        return result, output_files

    def _detect_video_boundaries(
        self, 
        video_path: str, 
        progress_callback=None
    ) -> List[float]:
        """
        检测视频边界（OCR标题画面）— 优先方法
        """
        video_boundaries = []
        
        try:
            from src.text_detector import TextFrameDetector
            detector = TextFrameDetector()
            
            def _video_progress(cur, msg):
                if progress_callback:
                    progress_callback(0.10 + 0.05 * cur, msg)
            
            video_boundaries = detector.find_title_boundaries(
                video_path,
                min_interval=45,  # 歌曲间隔至少45秒
                max_boundaries=15,
                progress_callback=_video_progress
            )
            self._ocr_title_frames = list(getattr(detector, "last_text_frames", []) or [])
            
            video_boundaries = [float(t) for t in video_boundaries]
            logger.info(f"OCR检测到 {len(video_boundaries)} 个标题画面")
            
        except ImportError as e:
            logger.warning(f"文字检测器未安装: {e}")
            self._ocr_title_frames = []
        except Exception as e:
            logger.warning(f"视频边界检测失败: {e}")
            self._ocr_title_frames = []
        
        return video_boundaries

    def _detect_audio_boundaries(
        self, 
        audio_path: str,
        progress_callback=None
    ) -> List[float]:
        """
        检测音频边界（能量谷值）— 辅助方法
        
        使用 soundfile 加载音频（避免 librosa.load 的路径/格式兼容问题）
        """
        audio_boundaries = []
        
        try:
            logger.info("开始 _detect_audio_boundaries...")
            
            from src.audio_analyzer import SongBoundaryDetector
            
            detector = SongBoundaryDetector(
                sample_rate=self.config.audio_sample_rate,
                hop_length=1024
            )
            
            # 使用 soundfile 加载音频（与 _create_songs 保持一致）
            import soundfile as sf
            import librosa as _librosa
            
            y, sr = sf.read(audio_path, dtype='float32')
            if len(y.shape) > 1:
                y = y.mean(axis=1)  # 转单声道
            
            total_duration = len(y) / sr
            
            # 如果采样率不匹配，重采样
            if sr != self.config.audio_sample_rate:
                y = _librosa.resample(y, orig_sr=sr, target_sr=self.config.audio_sample_rate)
                sr = self.config.audio_sample_rate
                total_duration = len(y) / sr
            
            logger.info(f"音频加载成功 (soundfile)，时长: {total_duration:.2f}秒")
            
            # 检测边界 - detect_boundaries 的回调是 (cur, total, msg)
            def audio_progress(cur, total, msg):
                if progress_callback:
                    progress_callback(0.15 + 0.05 * (cur / total if total > 0 else 0), msg)
            
            boundaries = detector.detect_boundaries(
                y, 
                total_duration,
                progress_callback=audio_progress
            )
            logger.info(f"detect_boundaries 完成，检测到 {len(boundaries)} 个边界")
            
            # 过滤太近的边界
            MIN_INTERVAL = 45.0
            filtered = []
            last = 0.0
            for b in sorted(boundaries):
                if b - last >= MIN_INTERVAL:
                    filtered.append(b)
                    last = b
            
            audio_boundaries = filtered
            logger.info(f"音频检测到 {len(audio_boundaries)} 个能量谷值边界")
            
        except Exception as e:
            logger.warning(f"音频边界检测失败: {e}")
        
        return audio_boundaries

    def _merge_boundaries(
        self,
        video_boundaries: List[float],
        audio_boundaries: List[float],
        msaf_boundaries: List[float],
        audio_path: str
    ) -> List[Tuple[float, float]]:
        """
        v3.0 Three-way merge: OCR + Audio Energy + MSAF structure analysis
        
        1. OCR boundaries = high-confidence anchors (title screens)
        2. Audio energy valleys = auxiliary (DISABLED - 误判太多)
        3. MSAF structure analysis = academic-level auxiliary (Foote etc.)
        """
        total_duration = self._get_audio_duration(audio_path)
        
        if video_boundaries:
            return self._build_songs_from_boundaries(video_boundaries, total_duration)
        
        # 禁用音频能量谷值边界，避免误判
        # aux_bounds = sorted(set(audio_boundaries + msaf_boundaries))
        aux_bounds = sorted(set(msaf_boundaries))
        if aux_bounds:
            return self._build_songs_from_boundaries(aux_bounds, total_duration)

        logger.warning("OCR/Audio/MSAF all empty -> whole video as one song")
        return [(0.0, total_duration)]

    def _detect_msaf_boundaries(
        self,
        audio_path: str,
        progress_callback=None
    ) -> List[float]:
        """MSAF music structure boundary detection (Foote / SF / OLDA algorithms)"""
        boundaries = []
        try:
            import scipy
            if not hasattr(scipy, "inf"):
                scipy.inf = np.inf
            import msaf
            logger.info("[MSAF] Starting music structure analysis...")
            try:
                results = msaf.process(audio_path, analyze=True)
            except TypeError:
                # MSAF 0.1.80 has no `analyze` kwarg.
                results = msaf.process(audio_path)
            if results is not None and len(results) >= 1:
                est_times = results[0]
                if est_times is not None and len(est_times) > 0:
                    for t in est_times:
                        t_float = float(t)
                        if t_float > 5.0:
                            boundaries.append(t_float)
                    logger.info(f"[MSAF] Found {len(boundaries)} structure boundaries")
                else:
                    logger.info("[MSAF] No structure boundaries found")
            else:
                logger.info("[MSAF] Empty result")
        except ImportError as e:
            logger.warning(f"[MSAF] Not installed, skipping: {e}")
        except Exception as e:
            logger.warning(f"[MSAF] Analysis failed: {e}")
        return boundaries

    def _smart_merge_3way(
        self,
        ocr_boundaries: List[float],
        audio_boundaries: List[float],
        msaf_boundaries: List[float],
        total_duration: float
    ) -> List[Tuple[float, float]]:
        """三路边界加权聚类融合（OCR 0.60 / Audio 0.25 / MSAF 0.15）。"""
        CLUSTER_RADIUS_SEC = 15.0
        KEEP_THRESHOLD = 0.55
        MIN_BOUNDARY_GAP = 60.0

        weighted_points: List[Tuple[float, str, float]] = []
        weighted_points.extend((float(t), "ocr", 0.60) for t in ocr_boundaries)
        weighted_points.extend((float(t), "audio", 0.25) for t in audio_boundaries)
        weighted_points.extend((float(t), "msaf", 0.15) for t in msaf_boundaries)
        weighted_points.sort(key=lambda x: x[0])

        clusters: List[dict] = []
        for t, source, w in weighted_points:
            matched = None
            for cluster in clusters:
                if abs(t - cluster["center"]) <= CLUSTER_RADIUS_SEC:
                    matched = cluster
                    break
            if matched is None:
                matched = {
                    "times": [],
                    "sources": set(),
                    "source_weights": {},
                    "weighted_sum": 0.0,
                    "total_weight": 0.0,
                    "center": t,
                }
                clusters.append(matched)

            matched["times"].append(t)
            matched["sources"].add(source)
            matched["source_weights"][source] = max(float(w), float(matched["source_weights"].get(source, 0.0)))
            matched["weighted_sum"] += t * w
            matched["total_weight"] += w
            matched["center"] = matched["weighted_sum"] / max(matched["total_weight"], 1e-8)

        candidates = []
        for cluster in clusters:
            score = sum(cluster["source_weights"].values())
            center = float(cluster["center"])
            if score >= KEEP_THRESHOLD and 0.0 < center < total_duration:
                candidates.append({"time": center, "score": score})

        # 距离过近时，保留 score 更高的候选边界
        candidates.sort(key=lambda x: x["time"])
        merged_candidates: List[dict] = []
        for cand in candidates:
            if not merged_candidates:
                merged_candidates.append(cand)
                continue
            if cand["time"] - merged_candidates[-1]["time"] < MIN_BOUNDARY_GAP:
                if cand["score"] > merged_candidates[-1]["score"]:
                    merged_candidates[-1] = cand
            else:
                merged_candidates.append(cand)

        final_boundaries = [c["time"] for c in merged_candidates]
        self._boundary_scores = {round(c["time"], 1): float(c["score"]) for c in merged_candidates}
        return self._build_songs_from_boundaries(final_boundaries, total_duration, min_duration=60.0)

    def _build_songs_from_boundaries(
        self,
        boundaries: List[float],
        total_duration: float,
        min_duration: float = 60.0
    ) -> List[Tuple[float, float]]:
        """从边界列表构建歌曲列表"""
        if not boundaries:
            return [(0.0, total_duration)]
        
        sorted_b = sorted(set(boundaries))
        # 加上首尾
        all_points = [0.0] + sorted_b + [total_duration]
        all_points = sorted(set(all_points))
        
        songs = []
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            if end - start >= min_duration:
                songs.append((start, end))
        
        # 至少返回一首
        if not songs:
            return [(0.0, total_duration)]

        return songs

    @staticmethod
    def _normalize_title_for_compare_legacy(text: str) -> str:
        if not text:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"[\(\[\{【（].*?[\)\]\}】）]", " ", text)
        text = re.sub(r"(live|演唱会|音乐会|歌词|官方|现场版|music|mv|karaoke)", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[^0-9a-z\u4e00-\u9fa5]+", "", text)
        return text

    @staticmethod
    def _sanitize_filename_component(text: str, fallback: str = "unknown") -> str:
        value = unicodedata.normalize("NFKC", str(text or "")).strip()
        value = re.sub(r"[\x00-\x1f]+", "", value)
        value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"_+", "_", value).strip(" ._")
        if not value:
            return fallback
        lowered = value.lower()
        if lowered in {"unknown", "n_a", "na", "null", "none", "???"}:
            return fallback
        if value.upper() in {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        }:
            value = f"{value}_"
        return value or fallback

    @staticmethod
    def _cosine_similarity(vec_a: Optional[Any], vec_b: Optional[Any]) -> float:
        if vec_a is None or vec_b is None:
            return 0.0
        try:
            a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
            b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
            if a.size == 0 or b.size == 0 or a.size != b.size:
                return 0.0
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1e-8 or nb <= 1e-8:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        except Exception:
            return 0.0
    
    def _run_doubao_asr(self, audio: np.ndarray, sr: int) -> dict:
        """
        使用豆包API进行语音识别
        
        Args:
            audio: 音频numpy数组
            sr: 采样率
            
        Returns:
            统一格式的ASR结果
        """
        import soundfile as sf
        import io
        
        # 将音频转换为WAV格式
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="wav")
        audio_data = buffer.getvalue()
        
        try:
            # 从环境变量获取豆包API配置
            appid = os.environ.get("DOUBAO_APPID", "6118416182")
            access_token = os.environ.get("DOUBAO_ACCESS_TOKEN", "wgYVCSXYek6ATuLNP_DiXFNHZ9jo5ZRV")
            
            doubao = DoubaoASR(appid=appid, access_token=access_token)
            doubao_result = doubao.recognize(
                audio_data=audio_data,
                language="zh-CN",
                caption_type="auto"
            )
            result = format_result(doubao_result)
        except Exception as e:
            result = {
                "engine": "doubao",
                "text": "",
                "words": [],
                "sentences": [],
                "language": "zh",
                "error": str(e)
            }
        
        return result

    def _compute_song_edge_signature(
        self,
        y_full: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float,
        edge: str,
        window_sec: float = 12.0,
    ) -> Optional[List[float]]:
        try:
            import librosa

            s = float(start_time)
            e = float(end_time)
            if e - s < 2.0:
                return None
            if edge == "head":
                clip_start = s
                clip_end = min(e, s + window_sec)
            else:
                clip_end = e
                clip_start = max(s, e - window_sec)
            y_clip = y_full[int(clip_start * sr):int(clip_end * sr)]
            if y_clip is None or len(y_clip) < int(sr * 2):
                return None
            chroma = librosa.feature.chroma_cqt(y=y_clip, sr=sr, hop_length=1024)
            if chroma is None or chroma.size == 0:
                return None
            signature = np.mean(chroma, axis=1).astype(np.float32)
            norm = float(np.linalg.norm(signature))
            if norm <= 1e-8:
                return None
            return (signature / norm).tolist()
        except Exception:
            return None

    def _attach_song_continuity_signatures(self, songs: List[SongInfo], y_full: np.ndarray, sr: int) -> None:
        for song in songs:
            try:
                head_sig = self._compute_song_edge_signature(
                    y_full=y_full,
                    sr=sr,
                    start_time=float(song.start_time),
                    end_time=float(song.end_time),
                    edge="head",
                )
                tail_sig = self._compute_song_edge_signature(
                    y_full=y_full,
                    sr=sr,
                    start_time=float(song.start_time),
                    end_time=float(song.end_time),
                    edge="tail",
                )
                setattr(song, "_head_sig", head_sig)
                setattr(song, "_tail_sig", tail_sig)
            except Exception:
                setattr(song, "_head_sig", None)
                setattr(song, "_tail_sig", None)

    @staticmethod
    def _segment_feature_value(seg: Segment, key: str, default: float = 0.0) -> float:
        feat = getattr(seg, "features", None)
        if feat is None:
            return default
        try:
            if isinstance(feat, dict):
                return float(feat.get(key, default))
            return float(getattr(feat, key, default))
        except Exception:
            return default

    @staticmethod
    def _clean_ocr_text(raw_text: str) -> str:
        text = str(raw_text or "").strip()
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"[|｜]+", " | ", text)
        return text[:120]

    def _is_likely_song_title_text(self, text: str) -> bool:
        """兼容包装：调用评分函数，阈值 >= 60 分通过"""
        score, reason = self._score_ocr_as_title(text)
        return score >= 60.0

    def _score_ocr_as_title(self, text: str) -> tuple:
        """
        A2 多维 OCR 歌名评分 (v2)
        
        Returns:
            (score: float, reason: str) — score ∈ [0, 100], reason 为得分/扣分说明
        """
        from typing import Tuple

        value = str(text or "").strip()
        reasons = []

        # ── 基础分 ──
        score = 50.0
        raw_len = len(value)

        # ═══ 加分项 ═══

        # 长度合理: 3-15 字符最像歌名
        if 3 <= raw_len <= 15:
            score += 20.0
            reasons.append(f"+长度合理({raw_len}字)")
        elif 16 <= raw_len <= 25:
            score += 8.0
            reasons.append(f"+长度偏长但可接受({raw_len}字)")
        else:
            pass  # 不加分不扣分（后面有长度惩罚）

        # 中文占比 > 60%
        cn_chars = re.findall(r"[\u4e00-\u9fa5]", value)
        alnum_chars = re.findall(r"[a-zA-Z0-9]", value)
        total_meaningful = len(cn_chars) + len(alnum_chars)
        if total_meaningful > 0 and len(cn_chars) / total_meaningful > 0.6:
            score += 10.0
            reasons.append(f"+中文主导")

        # 无标点符号
        punctuation_count = len(re.findall(r"[，。！？、；：""''【】《》()（）\[\].,!?;:\-]", value))
        if punctuation_count == 0:
            score += 8.0
            reasons.append("+无标点")

        # 不含歌词高频代词/虚词
        lyric_stopwords = {"爱", "想", "你", "我", "的", "了", "是", "在", "有",
                           "不", "要", "会", "能", "让", "把", "这", "那"}
        chars_in_text = set(value)
        lyric_overlap = len(chars_in_text & lyric_stopwords)
        # 如果歌词词只出现 1-2 个不算歌词，>=3 个才可疑
        if lyric_overlap <= 1:
            score += 5.0
            reasons.append("+非歌词风格")
        elif lyric_overlap >= 5:
            pass  # 后面扣分

        # 英文歌名模式：全英文且符合 Title Case 或全大写
        if value and all(c.isascii() and not c.isspace() or c == ' ' for c in value):
            stripped = value.replace(" ", "")
            if stripped.isalpha():
                if stripped[0].isupper():
                    score += 8.0
                    reasons.append("+英文Title Case模式")

        # ═══ 扣分项 ═══

        # 致命：长度极端
        if raw_len <= 2 or raw_len > 35:
            penalty = 40.0
            score -= penalty
            reasons.append(f"-长度异常({raw_len}字,-{penalty:.0f})")

        # 元数据关键词（致命）
        meta_keywords = r"(作曲|作词|lyrics|composed|copyright|字幕|制作|编曲|监制|出品|录音|混音|母带)"
        if re.search(meta_keywords, value, flags=re.IGNORECASE):
            penalty = 30.0
            score -= penalty
            matched_kw = re.search(meta_keywords, value, flags=re.IGNORECASE).group(1)
            reasons.append(f"-含元数据词'{matched_kw}'(-{penalty:.0f})")

        # 完整句子特征（两个以上句号/逗号等）
        sentence_puncts = re.findall(r"[，。！？！?;；]", value)
        if len(sentence_puncts) >= 2:
            penalty = 25.0
            score -= penalty
            reasons.append(f"-完整句子({len(sentence_puncts)}个标点,-{penalty:.0f})")

        # 现场互动词汇
        live_tokens = ("谢谢", "大家", "一起", "掌声", "欢呼", "安可", "今晚",
                       "现场", "我们", "你们", "喜欢", "开心", "感动", "加油")
        found_live = [t for t in live_tokens if t in value]
        if found_live:
            penalty = min(15.0 * len(found_live), 30.0)
            score -= penalty
            reasons.append(f"-现场词汇{found_live}(-{penalty:.0f})")

        # 歌词常见短语模式（强信号）
        lyric_patterns = (
            "我爱你", "在一起", "永远", "不想", "不要", "能不能",
            "好不好", "我知道", "你知道吗", "告诉我", "让我",
            "为你", "给我", "陪你", "想你", "爱你",
        )
        found_lyric = [p for p in lyric_patterns if p in value]
        if found_lyric:
            penalty = min(18.0 * len(found_lyric), 35.0)
            score -= penalty
            reasons.append(f"-歌词模式{found_lyric[:3]}(-{penalty:.0f})")

        # ── 动词重复/祈使句式歌词（v3 新增）──
        # 捕获 "来吧来吧去跳舞"、"走吧走吧不要回头" 等歌词
        # 特征：助词(吧/呀/哦/啊/嘛)+动词重复 或 动词+方向/动作
        imperative_patterns = [
            # 祈使句：动词+吧/呀/啊 重复出现（如"来吧来吧"、"走走走"）
            r"(?:来|去|走|跑|跳|飞|看|听|说|唱|喝|吃|睡|醒|起|坐|站|躺|笑|哭|喊|叫|抱|亲|吻|爱|恨|想|念|等|追|逃)[吧呀啊哦嘛]{1,2}(?:\s*[，,]\s*|\s*)+(?:来|去|走|跑|跳)",
            # 方向动词组合（来/去/回/进/出 + 动作动词）
            r"(?:来|去|回|进|出|上|下|前|后)\s*(?:跳|舞|跑|飞|唱|说|走|看|听|游|爬|转|摇|摆)",
            # 连续动词重复2次以上（如"走走走走"、"跑跑跑"）
            r"(.)\1{2,}",
            # 吧字句（如"来吧"、"去吧"、"走吧"、"跟我来吧"）
            r"\w+[吧呀嘛](?:\s*[,，]?\s*\w+[吧呀嘛])",
        ]
        matched_imperative = []
        for pat in imperative_patterns:
            if re.search(pat, value):
                matched_imperative.append(pat[:20])
        if len(matched_imperative) >= 1:
            penalty = min(22.0 + 12.0 * (len(matched_imperative) - 1), 40.0)
            score -= penalty
            reasons.append(f"-祈使句式歌词({len(matched_imperative)}个,-{penalty:.0f})")

        # ── 中文陈述句式歌词检测（v2 新增）──
        # 歌词常见结构：动词+了/着/过+宾语、无法/不能/可以+动词、副词+形容词
        lyric_sentence_patterns = [
            # 动词+时态助词+名词（如"忘了痛苦"、"忘记你的脸"、"感到思念"）
            r"(?:忘|记|想|看|听|说|做|爱|恨|怕|知|懂|等|找|走|来|去|回|留|放|拿|给|带|送|接|丢|碰|遇|见|变|成|变|让|使|叫|把|被)\s*[了着过]\s*[\u4e00-\u9fa5]",
            # 情态动词开头（无法/不能/可以/想要/愿意/应该）
            r"(?:无法|不能|可以|想要|愿意|应该|需要|希望|以为|觉得|感到|忽然|突然|依然|还是|总是|从来|也许|或许|如果|虽然|但是)\s*[\u4e00-\u9fa5]{2,}",
            # 主语+谓语+宾语的完整句式（如"我爱你"、"你爱我"——但短的不算）
            r"(?:我|你|他|她|它|我们|你们|他们|谁|什么|哪里|这|那|这里|那里)\s*(?:不\s*)?(?:爱|恨|想|要|会|能|知道|明白|记得|忘记|相信|担心|期待|等待|渴望|怀念|回忆|珍惜|守护|拥抱|亲吻|离开|放弃|坚持|努力|尝试|决定|答应|承诺|保证|发誓)\s*[\u4e00-\u9fa5]",
            # 形容词+的+情感名词（如"无比的思念"、"所有痛苦"）
            r"(?:无比|非常|特别|那么|这么|如此|多么|好|真|太|最|更|越|全|整|所有|一切|每个|任何)\s*(?:的|之)\s*[\u4e00-\u9fa5]{2,}",
            # 疑问/感叹句式（歌词常出现）
            r"(?:谁|什么|怎么|为什么|哪里|哪|几|多)\s*[\u4e00-\u9fa5]{2,}\s*(?:呢|吗|吧|啊|呀|喔|噢|啦)",
        ]
        matched_sentence_patterns = []
        for pat in lyric_sentence_patterns:
            if re.search(pat, value):
                matched_sentence_patterns.append(pat[:20])
        if len(matched_sentence_patterns) >= 1:
            # 句式匹配是强信号：至少命中1个句式就扣25分，每多一个加10分
            penalty = min(25.0 + 10.0 * (len(matched_sentence_patterns) - 1), 45.0)
            score -= penalty
            reasons.append(f"-陈述句式歌词({len(matched_sentence_patterns)}个,-{penalty:.0f})")

        # 词数过多
        tokens = value.split()
        if len(tokens) >= 6:
            penalty = 12.0 + 3 * (len(tokens) - 6)
            score -= min(penalty, 25.0)
            reasons.append(f"-词数过多({len(tokens)}个,-{min(penalty,25):.0f})")

        # 特殊字符过多
        special_chars = re.findall(r'[@_:;/\\&*#%$~^]', value)
        if len(special_chars) > 2:
            penalty = 20.0
            score -= penalty
            reasons.append(f"-特殊字符过多(-{penalty:.0f})")

        # 有效字符太少（全是符号）
        if raw_len > 0 and total_meaningful < 2:
            score -= 30.0
            reasons.append("-有效字符不足")

        # ═══ clamp & 返回 ═══
        final_score = max(0.0, min(100.0, round(score, 1)))
        reason_str = "|".join(reasons) if reasons else "(无特殊标记)"

        logger.debug(f"[OCR评分] '{value[:20]}' → {final_score}分 [{reason_str}]")

        return final_score, reason_str

    def _normalize_segment_label(self, label: str) -> str:
        """Normalize english/localized labels into canonical english tags."""
        if label is None:
            return ""
        raw = str(label).strip()
        if not raw:
            return ""

        raw_l = raw.lower()
        alias = {
            "pre-chorus": "prechorus",
            "instrumental": "inst",
            "solo": LABEL_SOLO,
            "speech": LABEL_SPEECH,
            "talk": LABEL_TALK,
            "audience": LABEL_AUDIENCE,
            "crowd": LABEL_CROWD,
        }
        raw_l = alias.get(raw_l, raw_l)

        canonical = {
            LABEL_SILENCE,
            LABEL_OTHER,
            LABEL_CHORUS,
            LABEL_CROWD,
            LABEL_AUDIENCE,
            LABEL_SPEECH,
            LABEL_TALK,
            LABEL_SOLO,
            LABEL_VERSE,
            LABEL_INTRO,
            LABEL_OUTRO,
            LABEL_INTERLUDE,
            "prechorus",
            "bridge",
            "inst",
        }
        if raw_l in canonical:
            return raw_l

        # SongFormer输出里 label 常是中文，借助 LABEL_CN 反查回英文标签。
        for en, local in (LABEL_CN or {}).items():
            local_text = str(local or "").strip()
            if not local_text:
                continue
            if raw == local_text or raw_l == local_text.lower():
                return str(en).strip().lower()

        return raw_l

    def _map_segment_label_to_export_type(self, label: str) -> Optional[str]:
        label_l = self._normalize_segment_label(label)
        if not label_l:
            return None
        if label_l in {LABEL_SILENCE, LABEL_OTHER}:
            return None
        if label_l in {LABEL_CHORUS}:
            return "副歌"
        if label_l in {LABEL_CROWD, LABEL_AUDIENCE}:
            return "合唱"
        if label_l in {LABEL_SPEECH, LABEL_TALK}:
            return "讲话串场"
        if label_l in {LABEL_SOLO, "inst"}:
            return "乐器SOLO"
        if label_l in {LABEL_CHORUS, "prechorus", "bridge"}:
            return "副歌"
        if label_l in {LABEL_VERSE, LABEL_INTRO, LABEL_OUTRO, LABEL_INTERLUDE}:
            return "主歌"
        return "主歌"

    def _is_highlight_segment(self, seg: Segment, export_type: str) -> bool:
        if export_type in {"副歌", "合唱"} and float(getattr(seg, "confidence", 0.0)) >= 0.55:
            return True
        raw_label = getattr(seg, "original_label", None) or getattr(seg, "label", "")
        if self._normalize_segment_label(raw_label) in {LABEL_CHORUS, LABEL_CROWD, LABEL_AUDIENCE}:
            return True
        relative_energy = self._segment_feature_value(seg, "relative_energy", 1.0)
        ssm_like = self._segment_feature_value(seg, "ssm_chorus_likelihood", 0.0)
        voice_ratio = self._segment_feature_value(seg, "voice_ratio", 0.0)
        return relative_energy >= 1.35 or ssm_like >= 0.65 or voice_ratio >= 0.9

    def _split_range_with_constraints(
        self,
        start: float,
        end: float,
        min_duration: float,
        max_duration: float,
    ) -> List[Tuple[float, float]]:
        duration = max(0.0, float(end) - float(start))
        if duration <= 0:
            return []
        if duration <= max_duration + 1e-6:
            return [(float(start), float(end))]

        pieces: List[Tuple[float, float]] = []
        n_parts = max(1, int(math.ceil(duration / max_duration)))
        cursor = float(start)
        remaining = duration

        for idx in range(n_parts):
            left_parts = n_parts - idx
            if left_parts <= 1:
                chunk = remaining
            else:
                low = max(min_duration, remaining - (left_parts - 1) * max_duration)
                high = min(max_duration, remaining - (left_parts - 1) * min_duration)
                if low > high:
                    low = max(0.0, remaining - (left_parts - 1) * max_duration)
                    high = min(max_duration, remaining)
                chunk = max(low, min(high, remaining / left_parts))
            next_cursor = min(float(end), cursor + max(0.01, chunk))
            pieces.append((round(cursor, 3), round(next_cursor, 3)))
            remaining = max(0.0, remaining - (next_cursor - cursor))
            cursor = next_cursor

        if pieces:
            pieces[-1] = (pieces[-1][0], float(end))
        return [(s, e) for s, e in pieces if e - s > 0.05]

    def _enforce_segment_safe_boundaries(
        self,
        segments: List[Dict[str, Any]],
        min_duration: float,
        max_duration: float,
    ) -> List[Dict[str, Any]]:
        """
        安全边界优先分段策略
        
        优先在以下位置切分：
        1. 字幕句末（如果有缓存的ASR结果）
        2. SongFormer段落边界
        3. 原始段落边界
        
        Args:
            segments: 原始段落列表
            min_duration: 最小时长
            max_duration: 最大时长
            
        Returns:
            优化后的段落列表
        """
        if not segments:
            return []
        
        result = []
        
        # 收集安全边界（如果有可用的）
        safe_boundaries = set()
        
        # 1. 收集ASR字幕边界（如果启用了字幕且有缓存）
        if hasattr(self, '_cached_asr_results') and self._cached_asr_results:
            for song_idx, asr_data in self._cached_asr_results.items():
                sentences = asr_data.get('sentences', [])
                words = asr_data.get('words', [])
                
                # 句子起始/结束时间都是优先的安全边界
                for sent in sentences:
                    sent_start = float(sent.get('start', 0))
                    sent_end = float(sent.get('end', 0))
                    # 找到对应的歌曲的起始时间，把相对时间转成绝对时间
                    # 先找这首歌
                    song_start = 0.0
                    for seg in segments:
                        song = seg.get('song')
                        if hasattr(song, 'song_index') and song.song_index == song_idx:
                            song_start = float(getattr(song, 'start_time', 0))
                            break
                    global_start = sent_start + song_start
                    global_end = sent_end + song_start
                    safe_boundaries.add(round(global_start, 3))
                    safe_boundaries.add(round(global_end, 3))
        
        # 2. 收集原始段落边界（安全保底）
        for seg in segments:
            safe_boundaries.add(round(seg['start'], 3))
            safe_boundaries.add(round(seg['end'], 3))
        
        # 3. 排序边界点
        sorted_boundaries = sorted(safe_boundaries)
        
        # 现在开始分段
        for seg in segments:
            original_start = seg['start']
            original_end = seg['end']
            dur = original_end - original_start
            
            # 判断是否是副歌
            is_chorus = (seg.get('type') == '副歌' or
                        seg.get('is_highlight', False) or
                        'chorus' in seg.get('songformer_label', '').lower())
            
            # 策略1：副歌优先完整不切（如果时长在可接受范围）
            if is_chorus and dur <= max_duration * 1.1:
                result.append(seg)
                continue
            
            # 策略2：其他情况用安全边界切
            if dur <= max_duration:
                # 本来就不长，保持原样
                result.append(seg)
                continue
            
            # 必须切分：找最安全的切分点
            current_start = original_start
            while current_start < original_end - 0.5:
                # 理想结束时间
                ideal_end = min(current_start + max_duration, original_end)
                
                # 在允许的范围内找安全边界
                search_start = max(current_start + min_duration, ideal_end - max(0.5, min_duration/2))
                search_end = min(ideal_end + 1.0, original_end)
                
                # 找范围内的安全边界
                best_end = None
                min_dist = float('inf')
                
                # 在边界点中找最好的
                for b in sorted_boundaries:
                    if search_start <= b <= search_end:
                        # 优先选靠近理想结束位置的
                        dist = abs(b - ideal_end)
                        if dist < min_dist:
                            min_dist = dist
                            best_end = b
                
                # 如果没找到，就用理想结束位置（或放宽一点）
                if best_end is None:
                    best_end = ideal_end
                
                # 确保时长符合要求
                if best_end - current_start < min_duration:
                    best_end = current_start + min_duration
                
                # 保存这个分段
                new_seg = dict(seg)
                new_seg['start'] = current_start
                new_seg['end'] = best_end
                result.append(new_seg)
                
                # 更新起始位置
                current_start = best_end
        
        # 最后一步：合并太短的相邻段落（同类型）
        final = []
        for seg in result:
            if not final:
                final.append(seg)
                continue
            
            last = final[-1]
            if last['type'] == seg['type']:
                gap = seg['start'] - last['end']
                combined_dur = seg['end'] - last['start']
                if gap <= 2.0 and combined_dur <= max_duration:
                    # 合并
                    last['end'] = seg['end']
                    last['is_highlight'] = last.get('is_highlight', False) or seg.get('is_highlight', False)
                    continue
            
            final.append(seg)
        
        return final

    def _enforce_segment_duration_constraints(
        self,
        segments: List[Dict[str, Any]],
        min_duration: float,
        max_duration: float,
    ) -> List[Dict[str, Any]]:
        if not segments:
            return []

        segs = sorted(segments, key=lambda x: (x["start"], x["end"]))

        # 1) Split oversized segments.
        expanded: List[Dict[str, Any]] = []
        for seg in segs:
            parts = self._split_range_with_constraints(seg["start"], seg["end"], min_duration, max_duration)
            for s, e in parts:
                copied = dict(seg)
                copied["start"] = s
                copied["end"] = e
                expanded.append(copied)
        segs = sorted(expanded, key=lambda x: (x["start"], x["end"]))

        # 2) Merge too-short segments with adjacent same-type ones.
        i = 0
        while i < len(segs):
            seg = segs[i]
            dur = seg["end"] - seg["start"]
            if dur >= min_duration - 1e-6:
                i += 1
                continue

            merged = False
            prev_gap = seg["start"] - segs[i - 1]["end"] if i > 0 else 9999.0
            next_gap = segs[i + 1]["start"] - seg["end"] if i + 1 < len(segs) else 9999.0
            
            # 同类型合并：优先尝试（间隔 <= 2.0s）
            if i > 0 and segs[i - 1]["type"] == seg["type"] and prev_gap <= 2.0:
                merged_dur = seg["end"] - segs[i - 1]["start"]
                if merged_dur <= max_duration + 1e-6:
                    segs[i - 1]["end"] = max(segs[i - 1]["end"], seg["end"])
                    segs[i - 1]["is_highlight"] = bool(segs[i - 1]["is_highlight"] or seg["is_highlight"])
                    del segs[i]
                    merged = True
                    i = max(0, i - 1)
            elif i + 1 < len(segs) and segs[i + 1]["type"] == seg["type"] and next_gap <= 2.0:
                merged_dur = segs[i + 1]["end"] - seg["start"]
                if merged_dur <= max_duration + 1e-6:
                    segs[i + 1]["start"] = min(segs[i + 1]["start"], seg["start"])
                    segs[i + 1]["is_highlight"] = bool(segs[i + 1]["is_highlight"] or seg["is_highlight"])
                    del segs[i]
                    merged = True
                    i = max(0, i - 1)

            if not merged:
                # 跨类型合并：如果合并后能被良好拆分，才合并
                can_merge_prev = i > 0 and (prev_gap <= 2.5 or segs[i - 1]["end"] >= seg["start"] - 0.1)
                can_merge_next = i + 1 < len(segs) and (next_gap <= 2.5 or segs[i + 1]["start"] <= seg["end"] + 0.1)
                
                merged_into_prev = False
                merged_into_next = False
                
                # 评估合并到前面的影响
                if can_merge_prev:
                    merged_start = segs[i - 1]["start"]
                    merged_end = seg["end"]
                    merged_dur = merged_end - merged_start
                    # 模拟拆分，看最小片段是否满足要求
                    parts = self._split_range_with_constraints(merged_start, merged_end, min_duration, max_duration)
                    min_part = min((e - s for s, e in parts), default=9999.0)
                    if merged_dur <= max_duration + 1e-6 and min_part >= min_duration * 0.8:
                        merged_into_prev = True
                
                # 评估合并到后面的影响
                if can_merge_next and not merged_into_prev:
                    merged_start = seg["start"]
                    merged_end = segs[i + 1]["end"]
                    merged_dur = merged_end - merged_start
                    parts = self._split_range_with_constraints(merged_start, merged_end, min_duration, max_duration)
                    min_part = min((e - s for s, e in parts), default=9999.0)
                    if merged_dur <= max_duration + 1e-6 and min_part >= min_duration * 0.8:
                        merged_into_next = True
                
                if merged_into_prev:
                    segs[i - 1]["end"] = max(segs[i - 1]["end"], seg["end"])
                    segs[i - 1]["is_highlight"] = bool(segs[i - 1]["is_highlight"] or seg.get("is_highlight", False))
                    del segs[i]
                    merged = True
                    i = max(0, i - 1)
                elif merged_into_next:
                    segs[i + 1]["start"] = min(segs[i + 1]["start"], seg["start"])
                    segs[i + 1]["is_highlight"] = bool(segs[i + 1]["is_highlight"] or seg.get("is_highlight", False))
                    del segs[i]
                    merged = True
                    i = max(0, i - 1)
                else:
                    # 无法合并：直接删除短片段
                    del segs[i]

        # 3) Re-split if any merged segment exceeds max; keep only min-compliant.
        final_segments: List[Dict[str, Any]] = []
        for seg in sorted(segs, key=lambda x: (x["start"], x["end"])):
            for s, e in self._split_range_with_constraints(seg["start"], seg["end"], min_duration, max_duration):
                if e - s >= min_duration - 1e-6:
                    copied = dict(seg)
                    copied["start"] = s
                    copied["end"] = e
                    final_segments.append(copied)

        return sorted(final_segments, key=lambda x: (x["start"], x["end"]))

    def _build_export_segments_for_song(
        self,
        song: SongInfo,
        min_duration: float,
        max_duration: float,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for seg in sorted(song.segments, key=lambda x: (x.start_time, x.end_time)):
            raw_label = getattr(seg, "original_label", None) or getattr(seg, "label", "")
            export_type = self._map_segment_label_to_export_type(raw_label)
            if not export_type:
                continue
            sf_label = getattr(seg, "songformer_label", "")
            logger.info(f"[Export DEBUG] segment: {seg.start_time:.1f}-{seg.end_time:.1f} label={seg.label} sf_label={sf_label}")
            candidates.append(
                {
                    "song": song,
                    "segment": seg,
                    "start": float(seg.start_time),
                    "end": float(seg.end_time),
                    "type": export_type,
                    "is_highlight": self._is_highlight_segment(seg, export_type),
                    "song_title": getattr(song, "song_title", "") or song.song_name,
                    "songformer_label": sf_label,
                }
            )
        logger.info(f"[Export DEBUG] candidates before rebalance: {len(candidates)}")
        for c in candidates[:3]:
            logger.info(f"[Export DEBUG] candidate: {c['start']:.1f}-{c['end']:.1f} sf_label={c['songformer_label']}")
        candidates = self._rebalance_song_export_types(candidates)
        # 使用安全边界优先策略
        result = self._enforce_segment_safe_boundaries(candidates, min_duration, max_duration)
        logger.info(f"[Export DEBUG] result after enforce: {len(result)}")
        for r in result[:3]:
            logger.info(f"[Export DEBUG] result: {r['start']:.1f}-{r['end']:.1f} sf_label={r.get('songformer_label', '')}")
        return result
    
    def _generate_subtitles_with_doubao(self, video_path: str, audio_segment: np.ndarray, sr: int, output_path: str) -> Tuple[bool, str]:
        """使用豆包 ASR 生成字幕并烧录到视频。"""
        import subprocess
        import tempfile

        output_spec = self._resolve_output_spec(video_path)
        orientation = output_spec.orientation
        asr_result = self._run_doubao_asr(audio_segment, sr)
        if asr_result.get("error"):
            return False, f"ASR识别失败: {asr_result['error']}"

        temp_dir = tempfile.gettempdir()
        ass_path = os.path.join(temp_dir, f"temp_subtitle_{uuid.uuid4().hex}.ass")
        from src.asr_subtitle import generate_ass_from_sentences, generate_ass_from_words

        if asr_result.get("sentences"):
            generate_ass_from_sentences(asr_result["sentences"], ass_path, output_spec=output_spec, orientation=orientation)
        elif asr_result.get("words"):
            generate_ass_from_words(asr_result["words"], ass_path, output_spec=output_spec, orientation=orientation)
        else:
            return False, "没有可用的识别文本"

        video_filter = build_cover_crop_filter(
            output_spec.width,
            output_spec.height,
            extra_filter=build_ass_filter_value(ass_path),
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        same_output_path = os.path.abspath(video_path) == os.path.abspath(output_path)
        working_output_path = (
            os.path.join(temp_dir, f"temp_subtitled_{uuid.uuid4().hex}.mp4")
            if same_output_path else output_path
        )

        def _build_cmd(video_codec_args: List[str], out_path: str) -> List[str]:
            return [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", video_filter,
                *video_codec_args,
                *STANDARD_VIDEO_OUTPUT_ARGS,
                "-c:a", "aac", "-b:a", "192k", "-ac", "2",
                "-r", str(self.config.export_fps),
                "-movflags", "+faststart",
                out_path,
            ]

        def _run_cmd(cmd: List[str]):
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=300,
            )

        try:
            run_result = _run_cmd(_build_cmd(
                ["-c:v", "h264_nvenc", "-rc", "cbr", "-b:v", "8M", "-maxrate", "8M", "-bufsize", "16M", "-preset", "p4"],
                working_output_path,
            ))
            if run_result.returncode != 0:
                logger.warning(f"NVENC失败，回退CPU: {run_result.stderr[-500:]}")
                run_result = _run_cmd(_build_cmd(
                    ["-c:v", "libx264", "-crf", "18", "-preset", "fast"],
                    working_output_path,
                ))
            if run_result.returncode != 0:
                return False, f"FFmpeg错误: {run_result.stderr[-500:]}"

            if same_output_path:
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(working_output_path, output_path)
            return True, output_path
        except subprocess.TimeoutExpired:
            return False, "FFmpeg处理超时(5分钟)"
        except Exception as e:
            return False, f"烧录字幕异常: {e}"
        finally:
            if os.path.exists(ass_path):
                try:
                    os.remove(ass_path)
                except Exception:
                    pass
            if same_output_path and os.path.exists(working_output_path):
                try:
                    os.remove(working_output_path)
                except Exception:
                    pass

    def _burn_subtitles_to_full_video(
        self,
        video_path: str,
        output_path: str,
        songs: List[SongInfo],
    ) -> Optional[str]:
        """将缓存 ASR 汇总后烧录到整条视频。"""
        import subprocess

        if not self._cached_asr_results:
            logger.warning("[字幕] 没有 ASR 缓存，跳过整条字幕烧录")
            return None

        all_sentences = []
        all_words = []
        for song in songs:
            cached = self._cached_asr_results.get(song.song_index)
            if not cached or cached.get("error"):
                continue
            for s in cached.get("sentences", []):
                adjusted = dict(s)
                adjusted["start"] = float(s["start"]) + float(song.start_time)
                adjusted["end"] = float(s["end"]) + float(song.start_time)
                all_sentences.append(adjusted)
            for w in cached.get("words", []):
                adjusted = dict(w)
                adjusted["start"] = float(w["start"]) + float(song.start_time)
                adjusted["end"] = float(w["end"]) + float(song.start_time)
                all_words.append(adjusted)

        if not all_sentences and not all_words:
            logger.warning("[字幕] 没有有效字幕，跳过整条字幕烧录")
            return None

        output_spec = self._resolve_output_spec(video_path)
        orientation = output_spec.orientation
        from src.asr_subtitle import generate_ass_from_sentences, generate_ass_from_words

        ass_path = os.path.join(tempfile.gettempdir(), f"temp_full_subtitle_{uuid.uuid4().hex}.ass")
        if all_sentences:
            generate_ass_from_sentences(all_sentences, ass_path, output_spec=output_spec, orientation=orientation)
        else:
            generate_ass_from_words(all_words, ass_path, output_spec=output_spec, orientation=orientation)

        video_filter = build_cover_crop_filter(
            output_spec.width,
            output_spec.height,
            extra_filter=build_ass_filter_value(ass_path),
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        def _build_cmd(video_codec_args, out_path):
            return [
                "ffmpeg", "-y", "-i", video_path,
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", video_filter,
                *video_codec_args,
                *STANDARD_VIDEO_OUTPUT_ARGS,
                "-c:a", "aac", "-b:a", "192k", "-ac", "2",
                "-r", str(self.config.export_fps), "-movflags", "+faststart",
                out_path,
            ]

        try:
            run_result = subprocess.run(
                _build_cmd(["-c:v", "h264_nvenc", "-rc", "cbr", "-b:v", "8M", "-maxrate", "8M", "-bufsize", "16M", "-preset", "p4"], output_path),
                capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=600
            )
            if run_result.returncode != 0:
                logger.warning(f"NVENC失败，回退CPU: {run_result.stderr[-500:]}")
                run_result = subprocess.run(
                    _build_cmd(["-c:v", "libx264", "-crf", "18", "-preset", "fast"], output_path),
                    capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=600
                )
            if run_result.returncode != 0:
                logger.error(f"FFmpeg字幕烧录失败: {run_result.stderr[-500:]}")
                return None
            logger.info(f"[字幕] 整条烧录完成: {output_path}")
            return output_path
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg字幕烧录超时(10分钟)")
            return None
        except Exception as e:
            logger.error(f"字幕烧录异常: {e}")
            return None
        finally:
            if os.path.exists(ass_path):
                try:
                    os.remove(ass_path)
                except OSError:
                    pass

    def _generate_subtitles_from_cached_asr(
        self,
        video_path: str,
        asr_result: dict,
        seg_start: float,
        seg_end: float,
        output_path: str,
        output_spec: Optional[OutputResolutionSpec] = None,
    ) -> Tuple[bool, str]:
        """从缓存 ASR 裁出片段字幕并烧录到片段视频。"""
        import subprocess
        import tempfile

        if output_spec is None:
            output_spec = self._resolve_output_spec(video_path)
        orientation = output_spec.orientation

        seg_sentences = []
        for s in asr_result.get("sentences", []):
            s_start = float(s.get("start", 0))
            s_end = float(s.get("end", 0))
            if s_end > seg_start and s_start < seg_end:
                seg_sentences.append({
                    "text": s.get("text", ""),
                    "start": max(0.0, s_start - seg_start),
                    "end": min(seg_end - seg_start, s_end - seg_start),
                })

        seg_words = []
        for w in asr_result.get("words", []):
            w_start = float(w.get("start", 0))
            w_end = float(w.get("end", 0))
            if w_end > seg_start and w_start < seg_end:
                seg_words.append({
                    "word": w.get("word", ""),
                    "start": max(0.0, w_start - seg_start),
                    "end": min(seg_end - seg_start, w_end - seg_start),
                })

        if not seg_sentences and not seg_words:
            return False, "该时间段无字幕"

        temp_dir = tempfile.gettempdir()
        ass_path = os.path.join(temp_dir, f"temp_subtitle_{uuid.uuid4().hex}.ass")
        from src.asr_subtitle import generate_ass_from_sentences, generate_ass_from_words
        if seg_sentences:
            generate_ass_from_sentences(seg_sentences, ass_path, output_spec=output_spec, orientation=orientation)
        else:
            generate_ass_from_words(seg_words, ass_path, output_spec=output_spec, orientation=orientation)

        video_filter = build_cover_crop_filter(
            output_spec.width,
            output_spec.height,
            extra_filter=build_ass_filter_value(ass_path),
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        same_output_path = os.path.abspath(video_path) == os.path.abspath(output_path)
        working_output_path = (
            os.path.join(temp_dir, f"temp_subtitled_{uuid.uuid4().hex}.mp4")
            if same_output_path else output_path
        )

        def _build_cmd(video_codec_args, out_path):
            return [
                "ffmpeg", "-y", "-i", video_path,
                "-map", "0:v:0", "-map", "0:a?",
                "-vf", video_filter,
                *video_codec_args,
                *STANDARD_VIDEO_OUTPUT_ARGS,
                "-c:a", "aac", "-b:a", "192k", "-ac", "2",
                "-r", str(self.config.export_fps), "-movflags", "+faststart",
                out_path,
            ]

        try:
            run_result = subprocess.run(
                _build_cmd(["-c:v", "h264_nvenc", "-rc", "cbr", "-b:v", "8M", "-maxrate", "8M", "-bufsize", "16M", "-preset", "p4"], working_output_path),
                capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=300
            )
            if run_result.returncode != 0:
                logger.warning(f"NVENC失败，回退CPU: {run_result.stderr[-500:]}")
                run_result = subprocess.run(
                    _build_cmd(["-c:v", "libx264", "-crf", "18", "-preset", "fast"], working_output_path),
                    capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=300
                )
            if run_result.returncode != 0:
                return False, f"FFmpeg错误: {run_result.stderr[-500:]}"

            if same_output_path:
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(working_output_path, output_path)
            return True, output_path
        except subprocess.TimeoutExpired:
            return False, "FFmpeg处理超时(5分钟)"
        except Exception as e:
            return False, f"烧录字幕异常: {e}"
        finally:
            if os.path.exists(ass_path):
                try:
                    os.remove(ass_path)
                except Exception:
                    pass
            if same_output_path and os.path.exists(working_output_path):
                try:
                    os.remove(working_output_path)
                except Exception:
                    pass

    def _rebalance_song_export_types(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reduce over-prediction of chorus in one song using lightweight guards."""
        if not segments:
            return segments

        type_counter = Counter(item["type"] for item in segments)
        total = len(segments)
        chorus_count = type_counter.get("副歌", 0)
        if total < 4 or chorus_count <= 0:
            return segments

        chorus_ratio = chorus_count / float(total)
        if chorus_ratio < 0.72:
            return segments

        adjusted: List[Dict[str, Any]] = []
        for item in segments:
            copied = dict(item)
            if copied["type"] == "副歌":
                seg = copied.get("segment")
                rel_energy = self._segment_feature_value(seg, "relative_energy", 1.0)
                ssm_like = self._segment_feature_value(seg, "ssm_chorus_likelihood", 0.0)
                confidence = float(getattr(seg, "confidence", 0.0) or 0.0)
                # weak-chorus => downgrade to verse
                if (rel_energy < 1.05 and ssm_like < 0.55) or confidence < 4.5:
                    copied["type"] = "主歌"
                    copied["is_highlight"] = False
            adjusted.append(copied)
        return adjusted

    def _extract_ocr_title_for_song_legacy(self, start_time: float, end_time: float) -> str:
        if not self._ocr_title_frames:
            return ""
        center = (start_time + end_time) / 2.0
        pool: Dict[str, float] = defaultdict(float)
        for item in self._ocr_title_frames:
            try:
                t = float(item.get("time", -1))
            except Exception:
                continue
            if t < start_time - 20 or t > min(end_time + 20, start_time + 90):
                continue
            raw_text = str(item.get("text", "")).strip()
            if len(raw_text) < 2:
                continue
            text = re.sub(r"[\r\n\t]+", " ", raw_text)
            text = re.sub(r"\s{2,}", " ", text).strip()
            text = re.split(r"[|｜/]+", text)[0].strip()
            score = float(item.get("score", 0.0) or 0.0)
            score += float(item.get("ocr_score", 0.0) or 0.0)
            score += max(0.0, 1.0 - abs(t - center) / 60.0)
            pool[text] += score
        if not pool:
            return ""
        best = max(pool.items(), key=lambda x: x[1])[0]
        return best[:60].strip()

    def _lookup_boundary_score(self, boundary_time: float) -> float:
        if not self._boundary_scores:
            return 0.0
        key = round(float(boundary_time), 1)
        if key in self._boundary_scores:
            return float(self._boundary_scores[key])
        nearest_key = min(self._boundary_scores.keys(), key=lambda k: abs(k - key))
        if abs(nearest_key - key) <= 0.6:
            return float(self._boundary_scores[nearest_key])
        return 0.0

    def _title_similarity(self, a: str, b: str) -> float:
        na = self._normalize_title_for_compare(a)
        nb = self._normalize_title_for_compare(b)
        if not na or not nb:
            return 0.0
        return float(SequenceMatcher(None, na, nb).ratio())

    def _identify_song_from_audio_clip(self, y_clip: np.ndarray, sr: int):
        if self._disable_song_identify:
            return None
        if y_clip is None or len(y_clip) < int(sr * 6):
            return None
        temp_wav = None
        try:
            import soundfile as sf
            from src.lyric_subtitle import identify_song_from_file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name
            sf.write(temp_wav, y_clip, sr)
            self._add_temp_file(temp_wav)

            result_holder = {"value": None, "error": None}

            def _runner():
                try:
                    result_holder["value"] = identify_song_from_file(temp_wav)
                except Exception as err:
                    result_holder["error"] = err

            # 防卡死：识曲超时后直接跳过，避免整个切片流程阻塞
            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join(timeout=12.0)
            if t.is_alive():
                logger.warning("song identify timed out (>12s), skip this clip")
                return None
            if result_holder["error"] is not None:
                raise result_holder["error"]
            return result_holder["value"]
        except Exception as e:
            logger.info(f"identify_song_from_file failed (non-blocking): {e}")
            return None

    def _identify_song_from_multiple_clips(
        self,
        song: SongInfo,
        y_full: np.ndarray,
        sr: int,
        max_probes: int = 3,
    ) -> Optional[Any]:
        """
        A1 增强版多片段识曲投票

        改进点（vs v1 简单多数投票）：
        1. 位置加权：中间段(0.5)权重 > 边缘段(0.2/0.8) —— 副歌通常在中间
        2. 置信度加权：高置信度的匹配票权更大
        3. 标题相似度聚类：无 track_id 时按标题相似度聚合而非精确匹配
        4. 更好的降级：无共识时返回最高置信结果，而不是直接丢弃
        """
        from difflib import SequenceMatcher as _SeqMatcher

        if y_full is None or sr <= 0 or song.duration < 8:
            return None

        song_start = float(song.start_time)
        song_end = float(song.end_time)
        song_duration = max(0.0, song_end - song_start)
        if song_duration <= 0:
            return None

        clip_len = min(20.0, max(8.0, song_duration * 0.35))
        center_ratios = (0.5, 0.2, 0.8)  # 先测中间段，命中后尽量不再额外调用识曲
        probe_budget = max(1, min(3, int(max_probes or 3)))
        probe_ratios = center_ratios[:probe_budget]
        position_weights = {0.2: 0.7, 0.5: 1.3, 0.8: 0.7}

        def _sample_match(ratio: float):
            center_val = song_start + song_duration * ratio
            clip_start = max(song_start, center_val - clip_len / 2.0)
            clip_end = min(song_end, clip_start + clip_len)
            clip_start = max(song_start, clip_end - clip_len)
            y_clip = y_full[int(clip_start * sr):int(clip_end * sr)]
            m = self._identify_song_from_audio_clip(y_clip, sr)
            if m and str(getattr(m, "title", "")).strip():
                setattr(m, "_pos_weight", float(position_weights.get(ratio, 1.0)))
                return m
            return None

        raw_matches: List[Any] = []
        first_match = _sample_match(probe_ratios[0])
        if first_match:
            first_tid = str(getattr(first_match, "track_id", "")).strip()
            first_conf = float(getattr(first_match, "confidence", 0.0) or 0.0)
            # 成本优化：中心片段命中且置信较高时直接返回，减少 2/3 外部识曲调用
            if first_tid or first_conf >= 0.82:
                logger.info("[A1投票] fast accept center clip: '%s' (conf=%.2f, tid=%s)",
                            str(getattr(first_match, "title", ""))[:40], first_conf, first_tid or "-")
                return first_match
            raw_matches.append(first_match)

        for ratio in probe_ratios[1:]:
            match = _sample_match(ratio)
            if match:
                raw_matches.append(match)

        if not raw_matches:
            return None

        # 第一轮：有 track_id 的硬匹配聚合
        id_matches = [(m, str(getattr(m, "track_id", "")).strip())
                      for m in raw_matches if getattr(m, "track_id", "").strip()]

        if len(id_matches) >= 2:
            # 有多个带 track_id 的结果，检查是否一致
            id_votes: Dict[str, list] = {}
            for m, tid in id_matches:
                id_votes.setdefault(tid, []).append(m)

            best_tid, group = max(id_votes.items(), key=lambda x: len(x[1]))
            if len(group) >= 2:
                # 2个以上片段确认同一首歌 => 高置信返回
                logger.info("[A1投票] %d/%d片段一致(track=%s)" % (len(group), len(raw_matches), best_tid))
                return self._pick_best_from_group(group)

        # 第二轮：无硬共识 => 加权打分选最优
        scored_candidates = []
        for m in raw_matches:
            title = str(getattr(m, "title", "")).strip()
            artist = str(getattr(m, "artist", "")).strip()
            conf = float(getattr(m, "confidence", 0.0) or 0.0)
            pos_w = float(getattr(m, "_pos_weight", 1.0))

            # 基础分 = 位置权重 * (1 + 置信度归一化)
            score = pos_w * (1.0 + min(conf, 1.0))

            # 与其他结果的相似度加分（一致性奖励）
            if len(raw_matches) > 1:
                sim_sum = sum(
                    _SeqMatcher(None, self._normalize_title_for_compare(title),
                                self._normalize_title_for_compare(
                                    str(getattr(other, "title", "")))).ratio() * 0.5
                    for other in raw_matches if other is not m
                )
                score += sim_sum / max(len(raw_matches) - 1, 1)

            scored_candidates.append((score, m, title, artist))

        # 按得分排序，取最高分
        scored_candidates.sort(key=lambda x: -x[0])
        best_score, best_match, best_title, best_artist = scored_candidates[0]

        second_score = scored_candidates[1][0] if len(scored_candidates) >= 2 else 0.0
        score_gap = best_score - second_score

        # 如果最高分和其他分差距太小(<15%)，说明各片段意见分歧大
        if len(scored_candidates) >= 2 and best_score > 0 and score_gap / best_score < 0.15:
            logger.info(
                "[A1投票] 分数差距过小(gap=%.2f/%.2f=%.0f%%), "
                "候选: %s" % (
                    score_gap, best_score, score_gap / best_score * 100,
                    [(s, t[:15]) for s, t, _a, _b in scored_candidates[:3]],
                )
            )
            # 即使有分歧也返回最高分的，但降低置信度作为信号
            orig_conf = float(getattr(best_match, "confidence", 0.0) or 0.0)
            setattr(best_match, "confidence", orig_conf * 0.6)
            setattr(best_match, "_vote_disagreement", True)
            return best_match

        logger.info("[A1投票] 加权选择: '%s' (score=%.2f, gap=%.2f)" % (best_title, best_score, score_gap))
        return best_match

    @staticmethod
    def _pick_best_from_group(group):
        """从一组匹配中选最佳（优先高置信+高位置权重）"""
        import logging
        logger = logging.getLogger(__name__)
        best = group[0]
        best_combined = float(getattr(best, "confidence", 0) or 0) * float(getattr(best, "_pos_weight", 1.0) or 1.0)
        for m in group[1:]:
            combined = float(getattr(m, "confidence", 0) or 0) * float(getattr(m, "_pos_weight", 1.0) or 1.0)
            if combined > best_combined:
                best = m
                best_combined = combined
        # 清理临时属性
        for m in group:
            for attr_name in ("_pos_weight",):
                if hasattr(m, attr_name):
                    delattr(m, attr_name)
        return best

    def _is_unknown_title(self, title: str) -> bool:
        t = (title or "").strip()
        if not t:
            return True
        return t in {"未知歌曲", "unknown", "Unknown", "N/A", "na", "NA"}

    def _is_title_quality_low(self, title: str) -> bool:
        text = str(title or "").strip()
        if self._is_unknown_title(text):
            return True
        if len(text) > 40:
            return True
        if re.search(r"(作曲|作词|lyrics|composed|copyright|字幕)", text, flags=re.IGNORECASE):
            return True
        if re.search(r"[_@;:]{2,}", text):
            return True
        return False
    
    def _identify_song_from_lyrics(self, lyrics: str, singer: str) -> Optional[Dict[str, str]]:
        """
        根据歌词和歌手名称识别歌名
        
        Args:
            lyrics: 识别的歌词文本
            singer: 歌手名称
            
        Returns:
            包含歌名和歌手的字典，或None
        """
        text = str(lyrics or "").strip()
        if not text:
            return None

        # 不再返回“未知歌曲”占位，避免覆盖音频识曲/OCR/提示词回退链路。
        # 这里仅做极轻量启发式：识别《歌名》样式，其他情况交由后续流程判断。
        try:
            for pattern in (r"《([^》]{2,30})》", r"\"([^\"]{2,30})\""):
                matched = re.search(pattern, text)
                if matched:
                    title = str(matched.group(1) or "").strip()
                    if title:
                        return {"title": title, "artist": singer or ""}
        except Exception as e:
            logger.warning(f"基于歌词的歌名识别失败: {e}")

        return None

    def _populate_song_identity(
        self,
        song: SongInfo,
        y_full: np.ndarray,
        sr: int,
        singer: Optional[str] = None,
    ) -> None:
        ocr_title = self._extract_ocr_title_for_song(song.start_time, song.end_time)
        song.boundary_confidence = self._lookup_boundary_score(song.start_time)

        # 步骤1: 尝试使用传统的音频识曲方法
        ocr_score, _reason = self._score_ocr_as_title(ocr_title) if ocr_title else (0.0, "")
        prefer_ocr_title = bool(ocr_title and ocr_score >= 72.0 and float(song.boundary_confidence or 0.0) >= 0.60)
        match = self._identify_song_from_multiple_clips(
            song,
            y_full,
            sr,
            max_probes=1 if prefer_ocr_title else 3,
        )

        audio_title = getattr(match, "title", "") if match else ""
        audio_title = str(audio_title or "").strip()
        audio_artist = str(getattr(match, "artist", "") if match else "").strip()
        ocr_title = str(ocr_title or "").strip()
        singer_norm = self._normalize_title_for_compare(singer or "")
        audio_artist_norm = self._normalize_title_for_compare(audio_artist)
        audio_artist_consistent = (
            not singer_norm
            or not audio_artist_norm
            or singer_norm == audio_artist_norm
            or singer_norm in audio_artist_norm
            or audio_artist_norm in singer_norm
        )

        # 步骤2: 提取音频片段并使用豆包API识别歌词
        lyrics = ""
        if not self._disable_lyrics_identify:
            try:
                # 提取歌曲中间部分的音频用于歌词识别
                song_start = float(song.start_time)
                song_end = float(song.end_time)
                song_duration = max(0.0, song_end - song_start)
                if song_duration > 10:
                    # 提取中间10秒的音频
                    clip_start = song_start + song_duration * 0.3
                    clip_end = min(song_end, clip_start + 10)
                    audio_segment = y_full[int(clip_start * sr):int(clip_end * sr)]
                    
                    # 使用豆包API识别歌词
                    asr_result = self._run_doubao_asr(audio_segment, sr)
                    lyrics = asr_result.get("text", "").strip()
                    logger.info(f"[Doubao] 识别歌词: {lyrics[:100]}...")
            except Exception as e:
                logger.warning(f"提取音频片段失败: {e}")
        else:
            logger.info(f"[歌词识别] 已禁用")

        # 步骤3: 根据歌词和歌手名称识别歌名
        lyrics_based_match = None
        if lyrics and singer:
            lyrics_based_match = self._identify_song_from_lyrics(lyrics, singer)
            if lyrics_based_match:
                logger.info(f"[Lyrics-based] 识别歌名: {lyrics_based_match.get('title', '未知歌曲')}")

        # 步骤4: 确定最终歌名
        final_title = "未知歌曲"
        
        # 优先级：1. 传统音频识曲 2. 基于歌词的识别 3. OCR标题
        if audio_title and not self._is_title_quality_low(audio_title) and audio_artist_consistent:
            final_title = audio_title
        elif lyrics_based_match:
            final_title = lyrics_based_match.get("title", "未知歌曲")
        elif ocr_title and (prefer_ocr_title or self._is_likely_song_title_text(ocr_title)):
            final_title = ocr_title
        elif audio_title:
            final_title = audio_title

        # 步骤5: 尝试从歌词提示文件中获取歌名
        if self._is_unknown_title(final_title):
            try:
                hint = self._load_song_hint(getattr(song, "song_name", ""), singer=singer)
                hint_title = str((hint or {}).get("title") or "").strip()
                hint_artist = str((hint or {}).get("artist") or "").strip()
                singer_norm = self._normalize_title_for_compare(singer or "")
                hint_artist_norm = self._normalize_title_for_compare(hint_artist)
                artist_match = (
                    bool(singer_norm)
                    and bool(hint_artist_norm)
                    and (
                        singer_norm == hint_artist_norm
                        or singer_norm in hint_artist_norm
                        or hint_artist_norm in singer_norm
                    )
                )
                if hint_title and artist_match:
                    final_title = hint_title
            except Exception:
                pass

        song.song_title = final_title.strip() or "未知歌曲"
        song.song_artist = (getattr(match, "artist", "") if match else "") or (lyrics_based_match.get("artist", "") if lyrics_based_match else "") or (singer or "")
        song.song_confidence = float(getattr(match, "confidence", 0.0) or 0.0)
        song.track_id = str(getattr(match, "track_id", "") or "")

        # keep OCR title for same-song merge rule (a)
        setattr(song, "_ocr_title", ocr_title.strip())
        setattr(song, "_ocr_title_norm", self._normalize_title_for_compare(ocr_title))

        if ocr_title and audio_title:
            sim = self._title_similarity(ocr_title, audio_title)
            logger.info(
                f"[song-id] OCR='{ocr_title}' audio='{audio_title}' similarity={sim:.2f} "
                f"(final='{song.song_title}', track_id='{song.track_id}')"
            )
        elif lyrics and final_title != "未知歌曲":
            logger.info(
                f"[song-id] Lyrics-based identification: '{final_title}' from lyrics: {lyrics[:100]}..."
            )

    def _should_merge_adjacent_songs(self, left: SongInfo, right: SongInfo) -> bool:
        """
        B1 增强版合并判断 — 加权打分 + 强信号门控

        不再使用 OR 链（任一条件触发就合并），而是：
        1. 各信号加权打分
        2. 必须有至少一个强信号(OCR完全匹配或track_id相同)
        3. 总分 >= 3.5 才合并
        """
        score = 0.0
        reasons = []

        # 强信号（每个可单独提供高权重）
        # (a) OCR 标准化标题完全匹配 -> 最强信号
        left_ocr = getattr(left, "_ocr_title_norm", "") or self._normalize_title_for_compare(getattr(left, "_ocr", ""))
        right_ocr = getattr(right, "_ocr_title_norm", "") or self._normalize_title_for_compare(getattr(right, "_ocr", ""))
        if left_ocr and right_ocr and left_ocr == right_ocr:
            score += 4.0
            reasons.append("ocr_exact")

        # (b) 同一首 track_id -> 最强信号
        if left.track_id and right.track_id and left.track_id == right.track_id:
            score += 4.0
            reasons.append("same_track")

        # 中等信号（需要累积）
        # (c) 标题相似度加权贡献
        sim = self._title_similarity(left.song_title, right.song_title)
        if sim > 0.60:
            sim_contribution = 2.5 * sim
            score += sim_contribution
            reasons.append(f"title_sim={sim:.2f}")

        boundary_score = self._lookup_boundary_score(left.end_time)
        gap = right.start_time - left.end_time
        audio_sim = self._cosine_similarity(
            getattr(left, "_tail_sig", None),
            getattr(right, "_head_sig", None),
        )
        if audio_sim >= 0.90:
            score += 3.5
            reasons.append(f"audio_cont={audio_sim:.2f}")
        elif audio_sim >= 0.84 and boundary_score < 0.65:
            score += 2.0
            reasons.append(f"audio_soft={audio_sim:.2f}")

        # (d) 双方都是未知歌名 + 弱边界 + 短间隔
        if self._is_unknown_title(left.song_title) and self._is_unknown_title(right.song_title):
            if gap <= 8.0 and boundary_score < 0.70:
                score += 2.0
                reasons.append("both_weak")
                if audio_sim >= 0.82:
                    score += 1.0
                    reasons.append("weak_boundary_audio")

        # 辅助弱信号（不可单独触发，但可辅助累积）
        # (e) 能量/声学特征相近（如果可用）
        if hasattr(left, 'avg_energy') and hasattr(right, 'avg_energy'):
            e_left = float(left.avg_energy or 0.0)
            e_right = float(right.avg_energy or 0.0)
            if e_left > 0.01 and e_right > 0.01:
                energy_ratio = min(e_left, e_right) / max(e_left, e_right)
                if energy_ratio > 0.75:
                    score += 1.0 * energy_ratio
                    reasons.append(f"energy_match={energy_ratio:.2f}")

        # (f) 短段 + 小间隔（保守：从30s提到40s，gap收窄到12s）
        left_dur = left.end_time - left.start_time
        right_dur = right.end_time - right.start_time
        gap = right.start_time - left.end_time

        if min(left_dur, right_dur) < 40.0 and gap < 12.0:
            score += 1.0
            reasons.append(f"short_segment(gap={gap:.1f}s)")

        # 门控判决
        has_strong_signal = any(r in ("ocr_exact", "same_track") for r in reasons)
        if any(str(r).startswith("audio_cont=") for r in reasons):
            has_strong_signal = True
        if (
            self._is_unknown_title(left.song_title)
            and self._is_unknown_title(right.song_title)
            and boundary_score < 0.62
            and audio_sim >= 0.88
        ):
            has_strong_signal = True
        should_merge = (score >= 3.5) and has_strong_signal

        logger.debug(
            "[合并判断] L='%s' R='%s' "
            "score=%.1f strong=%s merge=%s "
            "|%s|" % (
                left.song_title[:15], right.song_title[:15],
                score, has_strong_signal, "YES" if should_merge else "NO",
                ",".join(reasons),
            )
        )

        return should_merge

    def _merge_split_songs(self, songs: List[SongInfo]) -> List[SongInfo]:
        if len(songs) <= 1:
            return songs

        merged: List[SongInfo] = [songs[0]]
        for song in songs[1:]:
            prev = merged[-1]
            if self._should_merge_adjacent_songs(prev, song):
                prev.end_time = max(prev.end_time, song.end_time)
                prev.segments.extend(song.segments)
                prev.segments.sort(key=lambda s: (s.start_time, s.end_time))

                if self._is_unknown_title(prev.song_title) and not self._is_unknown_title(song.song_title):
                    prev.song_title = song.song_title
                prev.song_artist = prev.song_artist or song.song_artist
                prev.song_confidence = max(float(prev.song_confidence or 0.0), float(song.song_confidence or 0.0))
                if not prev.track_id and song.track_id:
                    prev.track_id = song.track_id
                prev.boundary_confidence = max(
                    float(prev.boundary_confidence or 0.0),
                    float(song.boundary_confidence or 0.0),
                )
            else:
                merged.append(song)

        for idx, song in enumerate(merged):
            song.song_index = idx
            song.song_name = f"Song_{idx+1:02d}"
            for seg in song.segments:
                seg.song_index = idx
        return merged

    @staticmethod
    def _normalize_title_for_compare(text: str) -> str:
        if not text:
            return ""
        value = str(text).lower().strip()
        value = re.sub(r"[\(\[\{<].*?[\)\]\}>]", " ", value)
        value = re.sub(r"(live|music|mv|karaoke|official|version|ver)", " ", value, flags=re.IGNORECASE)
        value = re.sub(r"[^0-9a-z\u4e00-\u9fa5]+", "", value)
        return value

    def _extract_ocr_title_for_song(self, start_time: float, end_time: float) -> str:
        if not self._ocr_title_frames:
            return ""
        center = (start_time + end_time) / 2.0
        pool: Dict[str, float] = defaultdict(float)
        for item in self._ocr_title_frames:
            try:
                t = float(item.get("time", -1))
            except Exception:
                continue
            if t < start_time - 20 or t > min(end_time + 20, start_time + 90):
                continue
            raw_text = str(item.get("text", "")).strip()
            if len(raw_text) < 2:
                continue
            text = self._clean_ocr_text(raw_text)
            chunks = [c.strip() for c in re.split(r"[|/]+", text) if c.strip()]
            if not chunks:
                chunks = [text]
            base_score = float(item.get("score", 0.0) or 0.0)
            base_score += float(item.get("ocr_score", 0.0) or 0.0)
            base_score += max(0.0, 1.0 - abs(t - center) / 60.0)
            for chunk in chunks[:4]:
                if not self._is_likely_song_title_text(chunk):
                    continue
                
                # v3 fix: 短文本更像歌名，长文本更可能是歌词/歌手名/组合信息
                # quality 范围约 [0.5, 1.5]：短歌名加分，长文本降权
                chunk_len = len(chunk)
                if chunk_len <= 4:
                    quality = 1.5          # 短歌名（如《一夜》《遇见》）最强候选
                elif chunk_len <= 10:
                    quality = 1.3          # 中等长度（标准歌名）
                elif chunk_len <= 18:
                    quality = 1.0          # 偏长（可能含副标题或歌手名）
                else:
                    quality = 0.6          # 很长（几乎肯定不是纯歌名）
                
                # OCR 置信度加成（高置信度的识别结果更可靠）
                ocr_conf = float(item.get("ocr_score", 0.0) or 0.0)
                ocr_bonus = 1.0 + ocr_conf * 0.5  # range [1.0, 1.5]
                
                pool[chunk] += base_score * quality * ocr_bonus
        if not pool:
            return ""
        return max(pool.items(), key=lambda x: x[1])[0][:60].strip()

    def _refine_segments_by_aed(
        self,
        segments: List[Any],
        singing_ts: List[Tuple[float, float]],
        speech_ts: List[Tuple[float, float]],
        song_start: float,
        song_end: float,
    ) -> List[Any]:
        """
        利用 FireRedAED 的 singing/speech 检测结果修正 SongFormer 段落分类。
        AED 输出: [(start_sec, end_sec), ...]，绝对时间。
        逻辑：
          - 如果段落在 singing 中度过高 → 增强 chorus/verse 置信度，抑制 speech 误判
          - 如果段落在 speech 中度过高但标为 verse → 降级为 speech/talk
        """
        if not segments:
            return segments

        def _aed_overlap_ratio(seg_start: float, seg_end: float, timestamps: List[Tuple[float, float]]) -> float:
            """计算某时间段被 AED timestamps 覆盖的比例"""
            if not timestamps:
                return 0.0
            total = seg_end - seg_start
            if total <= 0:
                return 0.0
            overlap = 0.0
            for a_start, a_end in timestamps:
                s = max(seg_start, a_start)
                e = min(seg_end, a_end)
                if e > s:
                    overlap += (e - s)
            return overlap / total

        refined = []
        changed = 0
        logger.info(f"[AED修正DEBUG] 开始处理 {len(segments)} 个段落")
        logger.info(f"[AED修正DEBUG] singing_ts={singing_ts}")
        logger.info(f"[AED修正DEBUG] speech_ts={speech_ts}")
        for seg in segments:
            seg_start, seg_end = seg.start_time, seg.end_time
            sing_ratio = _aed_overlap_ratio(seg_start, seg_end, singing_ts)
            speech_ratio = _aed_overlap_ratio(seg_start, seg_end, speech_ts)

            label = str(getattr(seg, 'label', '') or '').lower()
            new_label = label
            confidence = float(getattr(seg, 'confidence', 0.0) or 0.0)
            # 直接创建新字典，避免SegmentFeatures对象的问题
            features = {}

            logger.info(f"[AED修正DEBUG] [{seg_start:.1f}-{seg_end:.1f}s] label={label}, sing={sing_ratio:.0%}, speech={speech_ratio:.0%}")

            # 如果 singing 覆盖率高但标为 speech → 升为 verse
            if sing_ratio > 0.5 and label in (LABEL_SPEECH, LABEL_TALK, 'speech', 'talk'):
                new_label = LABEL_VERSE
                confidence = min(confidence + 0.3, 1.0)
                features['aed_singing_override'] = True
                changed += 1

            # 策略1：如果是 intro 标签，必须同时满足 speech_ratio>65% AND sing_ratio<45%
            if speech_ratio > 0.65 and sing_ratio < 0.45 and label in ('intro', '前奏'):
                new_label = LABEL_TALK
                confidence = min(confidence + 0.1, 1.0)
                features['aed_speech_override'] = True
                changed += 1
            # 策略2：如果是其他标签（verse/chorus/bridge），必须同时满足 speech_ratio>75% AND sing_ratio<25% 才修正
            elif speech_ratio > 0.75 and sing_ratio < 0.25 and label not in ('intro', '前奏'):
                new_label = LABEL_TALK
                confidence = min(confidence + 0.1, 1.0)
                features['aed_speech_override'] = True
                changed += 1

            if new_label != label:
                logger.info(f"[AED修正] [{seg_start:.1f}-{seg_end:.1f}s] {label} → {new_label} "
                             f"(sing={sing_ratio:.0%}, speech={speech_ratio:.0%})")

            refined.append(Segment(
                start_time=seg_start,
                end_time=seg_end,
                label=new_label,
                confidence=confidence,
                song_index=getattr(seg, 'song_index', 0),
                features=features,
                songformer_label=getattr(seg, 'songformer_label', ''),
            ))

        if changed > 0:
            logger.info(f"[AED修正] 共修正 {changed}/{len(segments)} 个段落")
        return refined

    def _create_songs(
        self,
        song_boundaries: List[Tuple[float, float]],
        audio_path: str,
        progress_callback=None,
        singer: Optional[str] = None,
        aed_wav_path: Optional[str] = None,
    ) -> List[SongInfo]:
        """
        创建歌曲结构，并对每首歌进行音频分析
        v7.0: 预计算歌曲级特征统计量，用于段落分类的相对能量基准（优化1）
        P1: enable_songformer=True 时使用 SongFormer SOTA 模型替代手工分类
        
        进度范围: 0.30 ~ 0.50 (占20个百分点)
        """
        songs = []
        total_songs = len(song_boundaries)
        song_progress_base = 0.30
        song_progress_range = 0.20
        global_stats: Dict[str, float] = {}
        fps_global = self.config.audio_sample_rate / 1024.0
        strict_songformer = bool(getattr(self.config, "strict_songformer", True))

        # 判断是否走手工分类路径
        use_manual_classification = not (
            self.config.enable_songformer and self.songformer_analyzer is not None
        )
        if strict_songformer and use_manual_classification:
            raise RuntimeError("SongFormer required but unavailable. 请先修复 SongFormer 模型加载。")

        # ── A. Demucs 人声分离（用 vocals 传给 AED，可大幅提升 singing 检测精度）──
        vocals_path: Optional[str] = None
        _demucs_tmp_dir: Optional[str] = None
        if self.config.enable_demucs:
            from src.audio_analyzer import separate_vocals
            try:
                import tempfile
                demucs_device = "cpu"
                try:
                    import torch
                    demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    demucs_device = "cpu"
                if demucs_device != "cuda":
                    logger.warning("[GPU] Demucs 未使用 CUDA，将以 CPU 运行。")
                self._log_gpu_memory("stage-B-demucs (pre)")
                _demucs_tmp_dir = tempfile.mkdtemp(prefix="dmucs_")
                vocals_path, _ = separate_vocals(
                    audio_path,
                    out_dir=_demucs_tmp_dir,
                    model=getattr(self.config, 'demucs_model', 'htdemucs_ft'),
                    device=demucs_device,
                )
                if vocals_path:
                    logger.info(f"[Demucs] 人声分离成功: {vocals_path}")
                else:
                    logger.warning("[Demucs] 人声分离失败，使用原始音频")
            except Exception as e:
                logger.warning(f"[Demucs] 人声分离出错: {e}")
                vocals_path = None
            finally:
                self._cleanup_gpu_stage("stage-B-demucs")

        # 仅手工分类路径才需要：加载全量音频 + 计算全局统计量
        if use_manual_classification:
            from src.audio_analyzer import SongBoundaryDetector, SegmentClassifier
            from scipy import ndimage

            # 使用 soundfile 替代 librosa.load()，避免加载问题
            try:
                import soundfile as sf
                import librosa
                y_full, sr = sf.read(audio_path, dtype='float32')  # 优化#5: float32 减半内存
                if len(y_full.shape) > 1:
                    y_full = y_full.mean(axis=1)  # 转单声道
                # 如果采样率不匹配，进行重采样
                if sr != self.config.audio_sample_rate:
                    y_full = librosa.resample(y_full, orig_sr=sr, target_sr=self.config.audio_sample_rate)
                    sr = self.config.audio_sample_rate
            except Exception as e:
                logger.error(f"加载音频失败: {e}")
                return songs

            fps_global = self.config.audio_sample_rate / 1024

            # 计算全局统计量
            rms_full = librosa.feature.rms(y=y_full, hop_length=1024)[0]
            global_stats = {
                'rms_mean': float(np.mean(rms_full)),
                'rms_p10': float(np.percentile(rms_full, 10)),
                'rms_p25': float(np.percentile(rms_full, 25)),
                'rms_p50': float(np.percentile(rms_full, 50)),
                'rms_p75': float(np.percentile(rms_full, 75)),
                'rms_p90': float(np.percentile(rms_full, 90)),
            }

            classifier = SegmentClassifier()
            classifier.set_global_stats(global_stats)
        else:
            # SongFormer 路径：只需加载音频用于切片，不计算特征
            try:
                import soundfile as sf
                import librosa
                y_full, sr = sf.read(audio_path, dtype='float32')  # 优化#5: float32
                if len(y_full.shape) > 1:
                    y_full = y_full.mean(axis=1)
                if sr != self.config.audio_sample_rate:
                    y_full = librosa.resample(y_full, orig_sr=sr, target_sr=self.config.audio_sample_rate)
                    sr = self.config.audio_sample_rate
            except Exception as e:
                logger.error(f"加载音频失败: {e}")
                return songs
            try:
                import librosa
                rms_full = librosa.feature.rms(y=y_full, hop_length=1024)[0]
                global_stats = {
                    'rms_mean': float(np.mean(rms_full)),
                    'rms_p10': float(np.percentile(rms_full, 10)),
                    'rms_p25': float(np.percentile(rms_full, 25)),
                    'rms_p50': float(np.percentile(rms_full, 50)),
                    'rms_p75': float(np.percentile(rms_full, 75)),
                    'rms_p90': float(np.percentile(rms_full, 90)),
                }
            except Exception:
                global_stats = {
                    'rms_mean': 0.05,
                    'rms_p10': 0.01,
                    'rms_p25': 0.02,
                    'rms_p50': 0.04,
                    'rms_p75': 0.07,
                    'rms_p90': 0.10,
                }

        for i, (start, end) in enumerate(song_boundaries):
            song_duration = end - start
            logger.info(f"分析歌曲 {i+1}: {start:.1f}s - {end:.1f}s (时长 {song_duration:.1f}s)")

            # 每首歌的进度
            if progress_callback and total_songs > 0:
                song_pct = i / total_songs
                p = song_progress_base + song_progress_range * song_pct
                progress_callback(p, f"[{i+1}/{total_songs}] 分析: {start:.0f}s - {end:.0f}s ...")

            # 提取歌曲音频片段
            y_song = y_full[int(start * sr):int(end * sr)]
            song_stats = self._compute_song_stats(y_song, sr, song_duration, start, fps_global)

            # ═══════════════════════════════════════════════════════════════
            # 路径选择：SongFormer SOTA 模型 vs 手工特征分类
            # ═══════════════════════════════════════════════════════════════
            if not use_manual_classification:
                # SongFormer 路径（SOTA 模型，跳过所有手工特征计算）
                if progress_callback:
                    song_min = int(start // 60)
                    song_sec = int(start % 60)
                    progress_callback(
                        song_progress_base + song_progress_range * (i + 0.3) / total_songs,
                        f"[{i+1}/{total_songs}] SongFormer GPU推理 ({song_min}:{song_sec:02d}, ~{song_duration:.0f}s) ..."
                    )
                try:
                    segments = self._analyze_song_segments_songformer(
                        y_song,
                        sr,
                        start,
                        song_duration,
                        global_stats=global_stats,
                        song_stats=song_stats,
                        song_index=i,  # W-3: 传入正确索引
                    )
                except Exception as sf_err:
                    logger.warning(f"[W-1] SongFormer 分析第{i+1}首失败: {sf_err}")
                    if strict_songformer:
                        raise RuntimeError(f"SongFormer failed on song {i+1}: {sf_err}") from sf_err
                    logger.info(f"[W-1] 自动降级到手工分类 (song {i+1})")
                    # 动态降级：这首歌及后续歌曲改用手工路径
                    if use_manual_classification is False:  # 还未降级过
                        use_manual_classification = True
                        # 初始化手工分类所需的依赖（延迟加载）
                        if 'classifier' not in dir():
                            from src.audio_analyzer import SegmentClassifier
                            classifier = SegmentClassifier()
                    # 用手工分类重试这首歌（确保 fps_global 已定义）
                    if progress_callback:
                        progress_callback(
                            song_progress_base + song_progress_range * (i + 0.5) / total_songs,
                            f"[{i+1}/{total_songs}] ⚠️ 降级为手工分析..."
                        )
                    # 安全：确保降级路径所需变量已定义（SongFormer 路径跳过了正常初始化）
                    if 'fps_global' not in dir():
                        fps_global = self.config.audio_sample_rate / 1024
                    if 'global_stats' not in dir():
                        rms_fallback = librosa.feature.rms(y=y_full, hop_length=1024)[0]
                        global_stats = {
                            'rms_mean': float(np.mean(rms_fallback)),
                            'rms_p10': float(np.percentile(rms_fallback, 10)),
                            'rms_p25': float(np.percentile(rms_fallback, 25)),
                            'rms_p50': float(np.percentile(rms_fallback, 50)),
                            'rms_p75': float(np.percentile(rms_fallback, 75)),
                            'rms_p90': float(np.percentile(rms_fallback, 90)),
                        }
                    segments = self._analyze_song_segments(
                        y_song, sr, start, song_duration,
                        classifier, global_stats, song_stats,
                    )
            else:
                # ═══════════════════════════════════════════════════════════════
                # 优化1+3+2：预计算歌曲级特征（相对能量 / 质心稳定性 / SSM似然）
                # ═══════════════════════════════════════════════════════════════
                # 在歌曲内部分析段落（传入歌曲级统计）
                segments = self._analyze_song_segments(
                    y_song,
                    sr,
                    start,
                    song_duration,
                    classifier,
                    global_stats,
                    song_stats,
                )

            if strict_songformer and not segments:
                raise RuntimeError(f"SongFormer returned no segments for song {i+1}.")

            # ── 接入 FireRedAED singing 检测，增强段落分类 ──────────────────────
            # AED 可区分 speech/singing/music，利用 singing 信号修正误判
            # 注意：AED 优先用原始音频（audio_path），不用Demucs分离后的vocals_path
            if self.audio_analyzer is not None and self.audio_analyzer.vad is not None:
                try:
                    if aed_wav_path:
                        # 使用全局预转换的WAV文件
                        current_aed_wav = aed_wav_path
                        logger.info(f"[AED] 使用全局预转换WAV: {os.path.basename(aed_wav_path)}")
                    else:
                        # 回退：每首歌转换一次
                        import tempfile
                        import subprocess
                        
                        tmp_aed_wav = tempfile.mktemp(suffix=".wav")
                        logger.info(f"[AED] 转换 {os.path.basename(audio_path)} -> WAV (回退)")
                        subprocess.run([
                            "ffmpeg", "-y", "-i", audio_path,
                            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", "-f", "wav", tmp_aed_wav
                        ], capture_output=True, check=True)
                        current_aed_wav = tmp_aed_wav
                    
                    logger.info(f"[AED] 开始检测 audio_path={current_aed_wav} (优先用原始音频，不用Demucs)")
                    aed_result = self.audio_analyzer.vad.get_aed_result(current_aed_wav)
                    if aed_result:
                        singing_ts = aed_result.get('event2timestamps', {}).get('singing', [])
                        speech_ts = aed_result.get('event2timestamps', {}).get('speech', [])
                        music_ts = aed_result.get('event2timestamps', {}).get('music', [])
                        logger.info(f"[AED] singing={len(singing_ts)}段, speech={len(speech_ts)}段, music={len(music_ts)}段")
                        logger.info(f"[AED] singing_ts={singing_ts[:3]}, speech_ts={speech_ts[:3]}")
                        if singing_ts or speech_ts:
                            segments = self._refine_segments_by_aed(
                                segments, singing_ts, speech_ts, start, end
                            )
                    else:
                        logger.warning(f"[AED] 返回空结果 audio_path={current_aed_wav}")
                    
                    # 清理回退情况下的临时文件（只在不是全局预转换的情况）
                    if not aed_wav_path:
                        try:
                            os.remove(current_aed_wav)
                        except:
                            pass
                except Exception as aed_err:
                    import traceback
                    logger.warning(f"FireRedAED 增强失败 (song {i+1}): {aed_err}")
                    logger.warning(f"FireRedAED traceback: {traceback.format_exc()}")

            song = SongInfo(
                song_index=i,
                song_name=f"Song_{i+1:02d}",
                segments=segments,
                start_time=start,
                end_time=end,
            )
            songs.append(song)
            
            # 歌曲完成进度
            if progress_callback and total_songs > 0:
                p = song_progress_base + song_progress_range * (i + 1) / total_songs
                progress_callback(p, f"[{i+1}/{total_songs}] {song.song_name} 完成 ({len(segments)}段)")

        for idx, song in enumerate(songs):
            if progress_callback and total_songs > 0:
                progress_callback(0.50, f"[识曲 {idx+1}/{total_songs}] 正在识别歌曲名...")

            # ── 字幕 + 歌名识别 ──
            if self.config.enable_subtitle and song.duration > 10:
                try:
                    # 对完整歌曲调豆包ASR，缓存结果供export复用
                    song_start = float(song.start_time)
                    song_end = float(song.end_time)
                    full_audio = y_full[int(song_start * sr):int(song_end * sr)]
                    asr_result = self._run_doubao_asr(full_audio, sr)
                    self._cached_asr_results[idx] = asr_result
                    logger.info(f"[Doubao] 歌曲{idx+1} ASR完成: {len(asr_result.get('sentences', []))}句")

                    # 用歌词+歌手匹配歌名
                    lyrics = asr_result.get("text", "").strip()
                    if lyrics and singer:
                        match = self._identify_song_from_lyrics(lyrics, singer)
                        if match:
                            song.song_title = match.get("title", "未知歌曲")
                            song.song_artist = match.get("artist", singer or "")
                            logger.info(f"[Doubao] 歌曲{idx+1} 识别歌名: {song.song_title}")
                            continue

                    logger.info(f"[Doubao] 歌曲{idx+1} 歌词未匹配歌名 (歌词前50字: {lyrics[:50]}...)")
                except Exception as e:
                    logger.warning(f"[Doubao] 歌曲{idx+1} ASR失败: {e}")

            # 默认：未知歌曲
            song.song_title = getattr(song, 'song_title', '') or "未知歌曲"
            song.song_artist = singer or ""
            song.song_confidence = 0.0
            song.track_id = ""

        self._attach_song_continuity_signatures(songs, y_full, sr)
        songs = self._merge_split_songs(songs)
        if not use_manual_classification:
            del y_full
            self._log_gpu_memory("stage-C-songformer")
            self._cleanup_gpu_stage("stage-C-songformer", unload_songformer=True)
        return songs

    def _compute_song_stats(
        self,
        y_song: np.ndarray,
        sr: int,
        song_duration: float,
        song_start: float,
        fps: float,
    ) -> dict:
        """
        预计算歌曲级统计量，供段落分类使用。

        包含（优化1）歌曲内相对能量基准、（优化3）质心稳定性、（优化2）SSM副歌似然。
        """
        import librosa
        from scipy import ndimage

        hop = 1024
        fps_song = sr / hop

        # ── 1. 歌曲内 RMS 统计（优化1）─────────────────────────────────
        rms_frames = librosa.feature.rms(y=y_song, hop_length=hop)[0]
        rms_smooth = ndimage.uniform_filter1d(rms_frames, size=int(fps_song * 3))
        song_rms_mean = float(np.mean(rms_smooth))
        song_rms_p50 = float(np.percentile(rms_smooth, 50))
        song_rms_p75 = float(np.percentile(rms_smooth, 75))
        song_rms_p85 = float(np.percentile(rms_smooth, 85))
        # 全曲质心统计（用于归一化）
        centroid_frames = librosa.feature.spectral_centroid(y=y_song, sr=sr, hop_length=hop)[0]
        centroid_smooth = ndimage.uniform_filter1d(centroid_frames, size=int(fps_song * 5))
        song_centroid_mean = float(np.mean(centroid_smooth))
        # 质心稳定性：每帧质心与平滑后质心的偏差，偏差越小越稳定
        centroid_dev = np.abs(centroid_frames - centroid_smooth)
        song_centroid_std = float(np.std(centroid_dev))
        song_centroid_stability = 1.0 / (1.0 + song_centroid_std / max(song_centroid_mean, 1))

        # ── 2. BPM 稳态（优化3）────────────────────────────────────────
        bpm_stability = 0.5  # 默认
        try:
            tempo, beats = librosa.beat.beat_track(y=y_song, sr=sr, hop_length=hop)
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
            if len(beat_times) > 3:
                intervals = np.diff(beat_times)
                bpm_stability = float(np.clip(1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8), 0, 1))
        except Exception:
            pass

        # ── 3. SSM 副歌似然（优化2）────────────────────────────────────
        # 对每帧计算其与所有其他帧的余弦相似度均值，
        # 作为"该帧属于副歌（重复结构）的概率"的简化近似
        ssm_chorus_likelihood_per_frame = np.zeros(len(rms_frames))
        try:
            mfcc = librosa.feature.mfcc(y=y_song, sr=sr, n_mfcc=13, hop_length=hop)
            chroma = librosa.feature.chroma_cqt(y=y_song, sr=sr, hop_length=hop)
            features_ssm = np.vstack([mfcc, chroma])
            # L2 归一化
            norms = np.linalg.norm(features_ssm, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            features_ssm = features_ssm / norms
            n = features_ssm.shape[1]
            # 对每帧，计算与前后窗口内所有帧的相似度均值
            win = max(1, int(5 * fps_song))  # 5秒窗口
            for j in range(n):
                w_start = max(0, j - win)
                w_end = min(n, j + win + 1)
                window = features_ssm[:, w_start:w_end]
                sims = np.dot(features_ssm[:, j], window)
                ssm_chorus_likelihood_per_frame[j] = float(np.mean(sims))
        except Exception as e:
            logger.debug(f"SSM 歌曲内分析失败: {e}")

        return {
            'song_rms_mean': song_rms_mean,
            'song_rms_p50': song_rms_p50,
            'song_rms_p75': song_rms_p75,
            'song_rms_p85': song_rms_p85,
            'song_centroid_mean': song_centroid_mean,
            'song_centroid_stability': song_centroid_stability,
            'bpm_stability': bpm_stability,
            'ssm_chorus_likelihood': ssm_chorus_likelihood_per_frame,  # [n_frames]
        }


    def _analyze_song_segments_songformer(
        self,
        y: np.ndarray,
        sr: int,
        song_start: float,
        song_duration: float,
        global_stats: Optional[dict] = None,
        song_stats: Optional[dict] = None,
        song_index: int = 0,  # B-3 修复: 传入正确歌曲索引
    ) -> List[Segment]:
        """
        P1: 使用 SongFormer SOTA 模型分析歌曲内段落

        B-1 修复：直接传入 numpy array 给 analyze_array()，
        跳过写临时文件 + librosa.load 的双重文件I/O。
        """
        from src.audio_analyzer import Segment

        # SongFormer 需要 24000Hz
        SONGFORMER_SR = 24000
        if sr != SONGFORMER_SR:
            import librosa as _librosa
            y_resampled = _librosa.resample(y, orig_sr=sr, target_sr=SONGFORMER_SR)
        else:
            y_resampled = y

        try:
            # B-1 修复：直接传数组，零文件 I/O
            segments_raw = self.songformer_analyzer.analyze_array(
                y_resampled.astype(np.float32), SONGFORMER_SR
            )
            logger.info(f"[SongFormer DEBUG] raw segments: {len(segments_raw)}")
            for s in segments_raw[:5]:
                logger.info(f"[SongFormer DEBUG] raw: {s['start']:.1f}-{s['end']:.1f} label={s['label']} original={s['original_label']}")
        except Exception as e:
            logger.error(f"[SongFormer] 分析失败 (song_index={song_index}): {e}")
            raise  # 由外层 W-1 的 try/except 捕获并降级

        # 转换为全局时间戳（song_start 偏移）
        segments = []
        for seg in segments_raw:
            start_global = song_start + seg['start']
            end_global = song_start + seg['end']

            # 过滤掉在歌曲范围外的段落
            if end_global <= song_start or start_global >= song_start + song_duration:
                continue

            # 截断到歌曲范围
            start_global = max(start_global, song_start)
            end_global = min(end_global, song_start + song_duration)

            if end_global - start_global < 8.0:  # 过滤过短段落
                continue

            segment = Segment(
                start_time=round(start_global, 2),
                end_time=round(end_global, 2),
                label=seg['label'],
                confidence=seg.get('confidence', 0.9),
                song_index=song_index,  # W-3 修复: 使用正确的索引
                features=None,
                songformer_label=seg.get('original_label', ''),
            )
            logger.info(f"[SongFormer DEBUG] created Segment: {start_global:.1f}-{end_global:.1f} songformer_label={segment.songformer_label}")
            segments.append(segment)

        segments = self._refine_songformer_segments_labels(
            y=y,
            sr=sr,
            song_start=song_start,
            song_duration=song_duration,
            segments=segments,
            global_stats=global_stats,
            song_stats=song_stats,
        )

        logger.info(f"[SongFormer DEBUG] after refine: {len(segments)} segments")
        for s in segments[:5]:
            logger.info(f"[SongFormer DEBUG] refined: {s.start_time:.1f}-{s.end_time:.1f} label={s.label} songformer_label={getattr(s, 'songformer_label', '')}")

        logger.info(f"[SongFormer] 分析完成: {len(segments)} 段")
        for s in segments:
            logger.info(
                f"  [{s.start_time:>7.1f}s - {s.end_time:>7.1f}s] "
                f"({s.duration:>5.1f}s) {s.label} (conf={s.confidence:.2f})"
            )

        return segments

    def _refine_songformer_segments_labels(
        self,
        y: np.ndarray,
        sr: int,
        song_start: float,
        song_duration: float,
        segments: List[Segment],
        global_stats: Optional[dict],
        song_stats: Optional[dict],
    ) -> List[Segment]:
        """Use acoustic classifier to correct SongFormer over-bias on chorus."""
        if not segments:
            return segments
        try:
            from src.audio_analyzer import SegmentClassifier
        except Exception:
            return segments

        gs = dict(global_stats or {})
        if not gs:
            try:
                import librosa
                rms_all = librosa.feature.rms(y=y, hop_length=1024)[0]
                gs = {
                    "rms_mean": float(np.mean(rms_all)),
                    "rms_p10": float(np.percentile(rms_all, 10)),
                    "rms_p25": float(np.percentile(rms_all, 25)),
                    "rms_p50": float(np.percentile(rms_all, 50)),
                    "rms_p75": float(np.percentile(rms_all, 75)),
                    "rms_p90": float(np.percentile(rms_all, 90)),
                }
            except Exception:
                gs = {"rms_mean": 0.05, "rms_p10": 0.01, "rms_p25": 0.02, "rms_p50": 0.04, "rms_p75": 0.07, "rms_p90": 0.10}

        ss = dict(song_stats or {})
        if not ss:
            try:
                ss = self._compute_song_stats(y, sr, song_duration, song_start, fps=sr / 1024.0)
            except Exception:
                ss = {}

        classifier = SegmentClassifier()
        classifier.set_global_stats(gs)

        refined: List[Segment] = []
        chorus_votes = 0
        for idx, seg in enumerate(segments):
            local_start = max(0.0, float(seg.start_time) - song_start)
            local_end = min(song_duration, float(seg.end_time) - song_start)
            y_seg = y[int(local_start * sr):int(local_end * sr)]
            if y_seg is None or len(y_seg) < 1024:
                refined.append(seg)
                continue

            features = self._extract_segment_features(y_seg, sr, sr / 1024.0, gs, ss)
            rel_energy = float(getattr(features, "relative_energy", 1.0) or 1.0)
            ssm_like = float(getattr(features, "ssm_chorus_likelihood", 0.0) or 0.0)
            context = {
                "position_ratio": (local_start / max(song_duration, 1e-8)),
                "song_duration": song_duration,
                "segment_index": idx,
                "total_segments": len(segments),
                "relative_energy": rel_energy,
                "ssm_chorus_likelihood": ssm_like,
            }
            scores = classifier._score_all(features, context)
            top_label = max(scores, key=scores.get)
            top_score = float(scores.get(top_label, 0.0))
            chorus_score = float(scores.get(LABEL_CHORUS, 0.0))
            verse_score = float(scores.get(LABEL_VERSE, 0.0))
            audience_score = float(scores.get(LABEL_AUDIENCE, 0.0))
            crowd_score = float(scores.get(LABEL_CROWD, 0.0))
            solo_score = float(scores.get(LABEL_SOLO, 0.0))
            
            logger.info(f"[REFINE-DEBUG] [{local_start:.1f}-{local_end:.1f}] orig_label={seg.label}, top={top_label}({top_score:.1f}), audience={audience_score:.1f}, crowd={crowd_score:.1f}, solo={solo_score:.1f}, chorus={chorus_score:.1f}, verse={verse_score:.1f}")

            new_label = str(seg.label)
            if top_label in {LABEL_SPEECH, LABEL_TALK, LABEL_CROWD, LABEL_AUDIENCE} and top_score >= 4.0:
                new_label = top_label
                logger.info(f"[REFINE-DEBUG] → 修正为 {top_label} (score {top_score:.1f} >=4.0)")
            elif top_label == LABEL_SOLO and top_score >= 3.5:
                new_label = LABEL_SOLO
                logger.info(f"[REFINE-DEBUG] → 修正为 {LABEL_SOLO} (score {top_score:.1f} >=3.5)")
            elif str(seg.label).lower() == LABEL_CHORUS and verse_score >= chorus_score + 1.2:
                new_label = LABEL_VERSE
            elif top_label == LABEL_CHORUS and chorus_score >= verse_score + 1.5:
                new_label = LABEL_CHORUS

            if new_label == LABEL_CHORUS:
                chorus_votes += 1

            refined.append(
                Segment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    label=new_label,
                    confidence=max(float(seg.confidence or 0.0), top_score),
                    song_index=seg.song_index,
                    features=features,
                    songformer_label=getattr(seg, "songformer_label", ""),
                )
            )

        # final guard: if chorus dominates, downcast weak chorus to verse
        if refined and chorus_votes / float(len(refined)) >= 0.5:
            for i, seg in enumerate(refined):
                if str(seg.label).lower() != LABEL_CHORUS:
                    continue
                rel_energy = self._segment_feature_value(seg, "relative_energy", 1.0)
                ssm_like = self._segment_feature_value(seg, "ssm_chorus_likelihood", 0.0)
                if rel_energy < 0.95 and ssm_like < 0.40:
                    refined[i] = Segment(
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        label=LABEL_VERSE,
                        confidence=seg.confidence,
                        song_index=seg.song_index,
                        features=seg.features,
                        songformer_label=getattr(seg, "songformer_label", ""),
                    )

        return refined

    def _analyze_song_segments(
        self,
        y: np.ndarray,
        sr: int,
        song_start: float,
        song_duration: float,
        classifier: 'SegmentClassifier',
        global_stats: dict,
        song_stats: dict,
    ) -> List[Segment]:
        """
        分析歌曲内的段落
        v7.0: 注入歌曲级统计（相对能量/质心稳定性/SSM似然）到上下文（优化1+2+3）
        """
        import librosa
        from scipy import ndimage

        fps = sr / 1024  # frames per second

        # ── 1. 能量包络（用于段落切割）────────────────────────────
        rms_frames = librosa.feature.rms(y=y, hop_length=1024)[0]
        rms_smooth = ndimage.uniform_filter1d(rms_frames, size=int(fps * 3))

        # ── 2. 质心（用于段落级质心梯度计算）──────────────────────
        centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=1024)[0]
        centroid_smooth = ndimage.uniform_filter1d(centroid_frames, size=int(fps * 5))

        # ── 3. 找能量谷值作为段落边界 ─────────────────────────────
        rms_threshold = np.mean(rms_smooth) * 0.7
        from scipy.signal import find_peaks
        inverted = -rms_smooth
        peaks, _ = find_peaks(inverted, distance=int(fps * 15))

        segment_starts = [0]
        for peak in peaks:
            if rms_smooth[peak] < rms_threshold:
                segment_starts.append(peak)
        segment_starts.append(len(rms_smooth))

        # ── 4. 逐段分类（注入歌曲级上下文）───────────────────────
        segments = []
        for i in range(len(segment_starts) - 1):
            start_frame = segment_starts[i]
            end_frame = segment_starts[i + 1]

            start_time = song_start + start_frame / fps
            end_time = song_start + end_frame / fps
            duration = end_time - start_time

            if duration < 8.0:
                continue

            y_seg = y[start_frame * 1024:end_frame * 1024]
            if len(y_seg) < 1024:
                continue

            # ── 歌曲级上下文计算 ───────────────────────────────────
            # 4a. 歌曲内相对能量
            seg_rms = float(np.sqrt(np.mean(y_seg ** 2)))
            song_rms_mean = song_stats.get('song_rms_mean', 0.01)
            relative_energy = seg_rms / max(song_rms_mean, 1e-8)

            # 4b. 质心稳定性和梯度（优化3）
            c_start = int(start_frame)
            c_end = min(int(end_frame), len(centroid_smooth))
            if c_end > c_start:
                seg_centroid = float(np.mean(centroid_smooth[c_start:c_end]))
            else:
                seg_centroid = song_stats.get('song_centroid_mean', 2000)
            seg_centroid_std = float(np.std(centroid_frames[c_start:c_end])) if c_end > c_start else 0
            centroid_stability = 1.0 / (1.0 + seg_centroid_std / max(song_stats.get('song_centroid_mean', 2000), 1))
            # 质心梯度：段首和段尾的差值（用于检测音色突变）
            seg_centroid_start = float(centroid_smooth[max(0, c_start)])
            seg_centroid_end = float(centroid_smooth[min(len(centroid_smooth) - 1, c_end - 1)])
            centroid_gradient = abs(seg_centroid_end - seg_centroid_start) / duration if duration > 0 else 0
            centroid_relative = seg_centroid / max(song_stats.get('song_centroid_mean', 2000), 1)

            # 4c. SSM 副歌似然（优化2）：取段落内所有帧的均值
            ssm_likelihood_frames = song_stats.get('ssm_chorus_likelihood', np.array([]))
            if len(ssm_likelihood_frames) > 0:
                f_start = max(0, start_frame)
                f_end = min(len(ssm_likelihood_frames), end_frame)
                if f_end > f_start:
                    ssm_chorus_likelihood = float(np.mean(ssm_likelihood_frames[f_start:f_end]))
                else:
                    ssm_chorus_likelihood = 0.0
            else:
                ssm_chorus_likelihood = 0.0
            # 归一化到 0~1（原始值范围大致 0.5~1.0）
            ssm_chorus_likelihood = max(0.0, min(1.0, (ssm_chorus_likelihood - 0.5) * 2.0))

            # ── 4d. 构造上下文（注入所有新特征）──────────────────
            position_ratio = (start_frame / fps) / song_duration if song_duration > 0 else 0.5
            context = {
                'position_ratio': position_ratio,
                'song_duration': song_duration,
                'segment_index': i,
                'total_segments': len(segment_starts) - 1,
                # 优化1：歌曲内相对能量基准
                'song_rms_baseline': song_rms_mean,
                'relative_energy': relative_energy,
                'song_rms_p50': song_stats.get('song_rms_p50', song_rms_mean),
                'song_rms_p75': song_stats.get('song_rms_p75', song_rms_mean),
                'song_rms_p85': song_stats.get('song_rms_p85', song_rms_mean),
                # 优化3：质心稳定性 + 梯度
                'centroid_stability': centroid_stability,
                'centroid_gradient': centroid_gradient,
                'centroid_relative': centroid_relative,
                # 优化3副：BPM 稳态
                'bpm_stability': song_stats.get('bpm_stability', 0.5),
                # 优化2：SSM 副歌似然
                'ssm_chorus_likelihood': ssm_chorus_likelihood,
            }

            # 计算段落特征（简化版，与 AudioAnalyzer 同步字段）
            features = self._extract_segment_features(
                y_seg, sr, fps, global_stats, song_stats
            )

            # v7.0 fix: 手动将 context 中的歌曲级特征注入 features（因为 _score_all 只读 features）
            features.song_rms_baseline = context.get('song_rms_baseline', 0.0)
            features.relative_energy = context.get('relative_energy', 1.0)
            features.song_rms_p50 = context.get('song_rms_p50', 0.0)
            features.song_rms_p75 = context.get('song_rms_p75', 0.0)
            features.song_rms_p85 = context.get('song_rms_p85', 0.0)
            features.centroid_stability = context.get('centroid_stability', 0.5)
            features.centroid_gradient = context.get('centroid_gradient', 0.0)
            features.centroid_relative = context.get('centroid_relative', 1.0)
            features.bpm_stability = context.get('bpm_stability', 0.5)
            features.ssm_chorus_likelihood = context.get('ssm_chorus_likelihood', 0.0)

            scores = classifier._score_all(features, context)
            label = max(scores, key=scores.get)

            # DEBUG: 记录第一个歌曲的前几个段落的分类详情
            if song_start < 10 and i < 3:
                from src.audio_analyzer import LABEL_CHORUS, LABEL_VERSE
                logger.info(
                    f"[DEBUG] Segment {i}: time={start_time:.1f}-{end_time:.1f}, "
                    f"label={label}, rel_energy={features.relative_energy:.2f}, "
                    f"ssm_like={features.ssm_chorus_likelihood:.2f}, "
                    f"centroid_stab={features.centroid_stability:.2f}, "
                    f"scores: chorus={scores.get(LABEL_CHORUS, 0):.1f}, verse={scores.get(LABEL_VERSE, 0):.1f}"
                )

            segment = Segment(
                start_time=start_time,
                end_time=end_time,
                label=label,
                confidence=scores[label],
                song_index=i,
            )
            segments.append(segment)

        return segments

    def _extract_segment_features(
        self,
        y: np.ndarray,
        sr: int,
        fps: float,
        global_stats: dict,
        song_stats: dict,
    ) -> 'SegmentFeatures':
        """
        提取段落特征（简化版本，用于快速分类）
        """
        import librosa
        from src.audio_analyzer import SegmentFeatures
        
        # 基础特征
        rms = float(np.sqrt(np.mean(y**2)))
        
        # 过零率
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=512)[0]))
        
        # 频谱质心
        try:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]))
        except:
            centroid = 2000
        
        # 频谱平坦度
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, hop_length=512)[0]))
        
        # 节拍特征
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_regularity = float(1.0 - onset_env.std() / (onset_env.mean() + 1e-6))
        beat_regularity = max(0.0, min(1.0, beat_regularity))
        
        # 人声比例（简化估计）
        voice_ratio = float(np.clip(rms / (global_stats.get('rms_mean', 0.1) + 1e-8), 0, 1))
        
        # 谐波比
        harmonic = librosa.effects.harmonic(y)
        harmonic_ratio = float(np.sqrt(np.mean(harmonic**2)) / (rms + 1e-8))
        
        # 静默比例
        silence_ratio = float(np.mean(np.abs(y) < 0.01))
        
        # 高频能量占比（>4kHz）
        try:
            spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
            n_bins = spec.shape[0]
            hf_bin = int(4000 / (sr / 1024))
            if hf_bin < n_bins:
                total_e = float(np.mean(spec ** 2))
                hf_e = float(np.mean(spec[hf_bin:, :] ** 2))
                hf_energy_ratio = hf_e / (total_e + 1e-8)
            else:
                hf_energy_ratio = 0.0
        except:
            hf_energy_ratio = 0.0

        # 创建特征对象（v7.0 补充歌曲级统计默认值，由 context 注入实际值）
        features = SegmentFeatures(
            rms=rms,
            rms_mean=global_stats.get('rms_mean', 0.1),
            rms_p10=global_stats.get('rms_p10', 0.01),
            rms_p15=global_stats.get('rms_p15', 0.02),
            rms_p25=global_stats.get('rms_p25', 0.03),
            rms_p50=global_stats.get('rms_p50', 0.05),
            rms_p60=global_stats.get('rms_p60', 0.06),
            rms_p75=global_stats.get('rms_p75', 0.08),
            rms_p90=global_stats.get('rms_p90', 0.1),
            zcr=zcr,
            centroid=centroid,
            flatness=flatness,
            silence_ratio=silence_ratio,
            beat_regularity=beat_regularity,
            voice_ratio=voice_ratio,
            harmonic_ratio=harmonic_ratio,
            chroma_entropy=3.0,  # 默认值
            spectral_flux=0.5,    # 默认值
            hf_zcr=0.05,          # 默认值
            spectral_spread=2000,  # 默认值
            delta_mfcc_energy=1.0,
            flatness_mean=flatness,
            hf_energy_ratio=hf_energy_ratio,
            # v7.0: 歌曲内相对能量（调用方通过 context 注入实际值）
            song_rms_baseline=0.0,
            relative_energy=1.0,
            song_rms_p50=0.0,
            song_rms_p75=0.0,
            song_rms_p85=0.0,
            centroid_stability=0.5,
            centroid_gradient=0.0,
            centroid_relative=1.0,
            bpm_stability=0.5,
            ssm_chorus_likelihood=0.0,
        )
        return features

    def _process_segments_legacy(
        self,
        video_path: str,
        analysis_result: AnalysisResult,
        progress_callback=None,
        singer: str = None
    ) -> List[str]:
        """处理所有片段"""
        from datetime import datetime
        from src.audio_analyzer import LABEL_CN
        
        output_files = []
        total_segments = sum(len(song.segments) for song in analysis_result.songs)
        processed = 0
        subtitle_audio_full = None
        subtitle_sr = 24000
        
        # 输出目录
        version_folder = datetime.now().strftime("%Y%m%d_%H%M")
        concert = self.config.concert or ""
        singer_safe = self._sanitize_filename_component(singer or "output", "output")
        concert_safe = self._sanitize_filename_component(concert or "", "")
        if concert:
            top_folder = f"{version_folder}_{singer_safe}_{concert_safe}"
        else:
            top_folder = f"{version_folder}_{singer_safe}"
        
        # 标签映射（简化：只保留用户需要的）
        SKIP_LABELS = {LABEL_SILENCE, LABEL_OTHER, LABEL_INTERLUDE}
        
        # 优化#4: 预获取视频信息，避免每个片段重复调用 ffprobe
        cached_video_info = self.ffmpeg_processor._get_video_info(video_path)

        if self.config.enable_subtitle:
            try:
                import librosa
                subtitle_audio_full, subtitle_sr = librosa.load(video_path, sr=subtitle_sr, mono=True)
            except Exception as e:
                logger.warning(f"[ZhengKai] preload subtitle audio failed, subtitles disabled: {e}")
                subtitle_audio_full = None
        
        for song in analysis_result.songs:
            song_dir = Path(self.config.output_dir) / top_folder / song.song_name
            song_dir.mkdir(parents=True, exist_ok=True)
            
            for seg in song.segments:
                # 跳过不需要的标签
                if seg.label in SKIP_LABELS:
                    continue
                
                duration = seg.end_time - seg.start_time
                if duration < self.config.min_segment_duration:
                    continue
                
                # 生成文件名
                label_cn = LABEL_CN.get(seg.label, seg.label)
                start_str = f"{int(seg.start_time//60):02d}m{int(seg.start_time%60):02d}s"
                end_str = f"{int(seg.end_time//60):02d}m{int(seg.end_time%60):02d}s"
                
                filename = f"{song.song_name}_{label_cn}_{start_str}-{end_str}.mp4"
                output_path = song_dir / filename
                
                # 切割视频（传入预缓存的 video_info 避免重复 ffprobe）
                cut_result = self.ffmpeg_processor.cut_video(
                    video_path,
                    seg.start_time,
                    seg.end_time,
                    str(output_path),
                    mode=self.config.cut_mode,
                    video_info=cached_video_info,  # 优化#4
                    output_spec=self._resolve_output_spec(video_path, cached_video_info),
                )
                if not cut_result.success:
                    logger.warning(
                        f"切片失败，已跳过: {output_path} - {cut_result.error_message}"
                    )
                    continue

                # ---- ASR subtitle (Doubao API) ----
                if self.config.enable_subtitle and subtitle_audio_full is not None:
                    subtitle_output_path = output_path.with_name(
                        f"{output_path.stem}_subtitled{output_path.suffix}"
                    )
                    audio_seg = subtitle_audio_full[
                        int(seg.start_time * subtitle_sr):int(seg.end_time * subtitle_sr)
                    ]
                    ok, res = self._generate_subtitles_with_doubao(
                        video_path=str(output_path),
                        audio_segment=audio_seg,
                        sr=subtitle_sr,
                        output_path=str(subtitle_output_path)
                    )
                    if not ok:
                        logger.warning(
                            f"[Doubao] subtitle failed: {res}"
                        )
                    else:
                        output_path = subtitle_output_path

                output_files.append(str(output_path))
                processed += 1
                 
                if progress_callback:
                    progress = 0.50 + 0.50 * (processed / max(1, total_segments))
                    progress_callback(progress, f"处理: {song.song_name} - {label_cn}")
        
        return output_files

    def _process_segments(
        self,
        video_path: str,
        analysis_result: AnalysisResult,
        progress_callback=None,
        singer: str = None
    ) -> List[str]:
        """Process all segments and export with normalized type/name rules."""
        output_files: List[str] = []
        self._last_export_metadata = []
        subtitle_audio_full = None
        subtitle_sr = 24000

        min_duration = float(getattr(self.config, "min_duration_limit", 8.0) or 8.0)
        max_duration = float(getattr(self.config, "max_duration_limit", 15.0) or 15.0)
        min_duration = max(8.0, min(20.0, min_duration))
        max_duration = max(8.0, min(20.0, max_duration))
        if max_duration < min_duration:
            max_duration = min_duration

        version_folder = datetime.now().strftime("%Y%m%d_%H%M")
        concert = self.config.concert or ""
        singer_safe = self._sanitize_filename_component(singer or "output", "output")
        concert_safe = self._sanitize_filename_component(concert or "", "")
        if concert_safe:
            top_folder = f"{version_folder}_{singer_safe}_{concert_safe}"
        else:
            top_folder = f"{version_folder}_{singer_safe}"

        cached_video_info = self.ffmpeg_processor._get_video_info(video_path)

        if self.config.enable_subtitle:
            try:
                import librosa
                subtitle_audio_full, subtitle_sr = librosa.load(video_path, sr=subtitle_sr, mono=True)
            except Exception as e:
                logger.warning(f"[subtitle] preload failed, disabled for this run: {e}")
                subtitle_audio_full = None

        export_queue: List[Dict[str, Any]] = []
        for song in analysis_result.songs:
            export_queue.extend(self._build_export_segments_for_song(song, min_duration, max_duration))

        total_segments = len(export_queue)
        processed = 0
        date_prefix = datetime.now().strftime("%Y%m%d")
        global_seq = 0

        for item in export_queue:
            song: SongInfo = item["song"]
            seg: Segment = item["segment"]
            export_type = item["type"]
            is_highlight = bool(item["is_highlight"])
            start_time = float(item["start"])
            end_time = float(item["end"])

            setattr(seg, "export_type", export_type)
            setattr(seg, "is_highlight", is_highlight)

            raw_song_title = "未知歌曲"
            song_folder = self._sanitize_filename_component(raw_song_title, "未知歌曲")
            song_dir = Path(self.config.output_dir) / top_folder / song_folder
            song_dir.mkdir(parents=True, exist_ok=True)

            song_title_for_name = self._sanitize_filename_component(raw_song_title, "未知歌曲")
            global_seq += 1
            seq = global_seq

            filename = f"{date_prefix}_{song_title_for_name}_{seq:03d}_{export_type}.mp4"
            output_path = song_dir / filename

            cut_result = self.ffmpeg_processor.cut_video(
                video_path,
                start_time,
                end_time,
                str(output_path),
                mode=self.config.cut_mode,
                video_info=cached_video_info,
                output_spec=self._resolve_output_spec(video_path, cached_video_info),
            )
            if not cut_result.success:
                logger.warning(f"clip failed, skipped: {output_path} - {cut_result.error_message}")
                continue

            if self.config.enable_subtitle and subtitle_audio_full is not None:
                subtitle_tmp = output_path.with_name(f"{output_path.stem}_subtitled{output_path.suffix}")
                audio_seg = subtitle_audio_full[int(start_time * subtitle_sr):int(end_time * subtitle_sr)]
                ok, res = self._generate_subtitles_with_doubao(
                    video_path=str(output_path),
                    audio_segment=audio_seg,
                    sr=subtitle_sr,
                    output_path=str(subtitle_tmp)
                )
                if not ok:
                    logger.warning(f"[Doubao] subtitle failed for {output_path.name}: {res}")
                else:
                    try:
                        shutil.move(str(subtitle_tmp), str(output_path))
                    except Exception as move_err:
                        logger.warning(f"[subtitle] finalize move failed: {move_err}")
                        output_path = subtitle_tmp

            output_files.append(str(output_path))
            processed += 1

            export_meta = {
                "path": str(output_path),
                "type": export_type,
                "highlight": is_highlight,
                "song_title": raw_song_title,
                "song_name": song.song_name,
                "start": start_time,
                "end": end_time,
            }
            self._last_export_metadata.append(export_meta)
            logger.info(
                f"[export] {output_path.name} | song={raw_song_title} | type={export_type} "
                f"| highlight={is_highlight} | {start_time:.2f}s-{end_time:.2f}s"
            )

            if progress_callback:
                progress = 0.50 + 0.50 * (processed / max(1, total_segments))
                progress_callback(progress, f"处理: {raw_song_title} - {export_type}")

        return output_files


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live演唱会视频智能切片')
    parser.add_argument('video', help='视频文件路径')
    parser.add_argument('-s', '--singer', help='歌手名称')
    parser.add_argument('-m', '--min-duration', type=float, default=8.0, help='最小片段时长(秒)')
    parser.add_argument('-o', '--output', default=r'D:\video_clip\output', help='输出目录')
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        output_dir=args.output,
        min_segment_duration=args.min_duration,
        min_duration_limit=args.min_duration,
    )
    
    processor = LiveVideoProcessor(config)
    result, files = processor.process_video(
        args.video,
        singer=args.singer
    )
    
    print(f"\n处理完成！")
    print(f"歌曲数: {len(result.songs)}")
    print(f"切片数: {len(files)}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
