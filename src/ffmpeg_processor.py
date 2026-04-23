"""
FFmpeg 视频处理模块
提供音频提取、视频切片、音频增强、视频转码和批量处理功能
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
import json

from src.output_spec import OutputResolutionSpec, build_cover_crop_filter, STANDARD_VIDEO_OUTPUT_ARGS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """处理结果数据类"""
    success: bool
    output_path: Optional[str]
    error_message: Optional[str]
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class FFmpegProcessor:
    """
    FFmpeg 视频处理器
    
    提供完整的音视频处理功能，包括:
    - 音频提取
    - 视频切片
    - 音频增强
    - 视频转码
    - 批量处理
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        初始化 FFmpeg 处理器
        
        Args:
            ffmpeg_path: FFmpeg 可执行文件路径，默认为 "ffmpeg"
            ffprobe_path: FFprobe 可执行文件路径，默认为 "ffprobe"
        """
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """检查 FFmpeg 是否可用"""
        try:
            result = subprocess.run(
                [self.ffmpeg, "-version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=True
            )
            version_line = result.stdout.split('\n')[0]
            logger.info(f"FFmpeg 版本: {version_line}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"FFmpeg 检查失败: {e}")
            raise RuntimeError(f"FFmpeg 不可用，请确保 FFmpeg 已安装并在 PATH 中: {e}")
    
    @staticmethod
    def _safe_json_loads(payload: Any) -> Dict[str, Any]:
        """安全解析 ffprobe JSON 输出。"""
        if payload is None:
            return {}
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", errors="ignore")
        if not isinstance(payload, str):
            return {}
        payload = payload.strip()
        if not payload:
            return {}
        try:
            data = json.loads(payload)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            logger.warning(f"ffprobe JSON 解析失败: {e}")
            return {}

    def _run_ffmpeg(self, args: List[str], progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """
        运行 FFmpeg 命令
        
        Args:
            args: FFmpeg 参数列表
            progress_callback: 进度回调函数，接收 (progress_percent, current_time, total_duration)
        
        Returns:
            (success, error_message)
        """
        cmd = [self.ffmpeg] + args
        logger.debug(f"执行命令: {' '.join(cmd)}")
        
        try:
            # Windows 下处理中文路径，禁用 universal_newlines，使用二进制模式
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 解析进度信息
            duration = None
            if progress_callback:
                # 尝试从输入文件获取总时长
                for arg in args:
                    if arg.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3')):
                        duration = self._get_duration(arg)
                        break
            
            stderr_output = []
            
            for line_bytes in process.stderr:
                try:
                    line = line_bytes.decode('utf-8', errors='ignore')
                except UnicodeDecodeError as e:
                    logger.warning(f"解码FFmpeg输出失败: {e}, 使用原始字节")
                    line = str(line_bytes)
                except Exception as e:
                    logger.error(f"处理FFmpeg输出时发生未知错误: {e}")
                    line = str(line_bytes)
                stderr_output.append(line)
                
                if progress_callback and duration:
                    # 解析时间进度
                    time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
                    if time_match:
                        hours = int(time_match.group(1))
                        minutes = int(time_match.group(2))
                        seconds = float(time_match.group(3))
                        current_time = hours * 3600 + minutes * 60 + seconds
                        progress = min(100, (current_time / duration) * 100)
                        progress_callback(progress, current_time, duration)
            
            process.wait()
            
            if process.returncode != 0:
                # 解码错误输出
                decoded_errors = []
                for err_line in stderr_output[-10:]:
                    try:
                        # 优化#13: 兼容 bytes 和 str 类型
                        if isinstance(err_line, bytes):
                            decoded_errors.append(err_line.decode('utf-8', errors='ignore'))
                        else:
                            decoded_errors.append(str(err_line))
                    except Exception as e:
                        logger.warning(f"解码错误输出失败: {e}, 使用字符串表示")
                        decoded_errors.append(str(err_line))
                error_msg = ''.join(decoded_errors)
                logger.error(f"FFmpeg 执行失败: {error_msg}")
                return False, error_msg
            
            return True, ""
            
        except Exception as e:
            logger.error(f"FFmpeg 执行异常: {e}")
            return False, str(e)
    
    def _get_duration(self, media_path: str) -> Optional[float]:
        """获取媒体文件时长"""
        try:
            cmd = [
                self.ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                media_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=True)
            data = self._safe_json_loads(result.stdout)
            duration_value = (data.get("format") or {}).get("duration")
            if duration_value in (None, "", "N/A"):
                raise ValueError("ffprobe 未返回 duration")
            return float(duration_value)
        except Exception as e:
            logger.warning(f"无法获取媒体时长: {e}")
            return None
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频信息（内部方法）"""
        try:
            cmd = [
                self.ffprobe,
                "-v", "error",
                "-show_entries", "format=duration:stream=width,height,codec_name",
                "-of", "json",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=True)
            data = self._safe_json_loads(result.stdout)
            if not data:
                logger.warning("ffprobe 输出为空或非 JSON")
                return {}

            format_info = data.get("format") or {}
            duration_value = format_info.get("duration")
            duration = 0.0
            if duration_value not in (None, "", "N/A"):
                try:
                    duration = float(duration_value)
                except (TypeError, ValueError):
                    duration = 0.0

            info = {
                'duration': duration,
                'width': None,
                'height': None,
                'codec': None
            }

            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video' or 'width' in stream:
                    info['width'] = stream.get('width')
                    info['height'] = stream.get('height')
                    info['codec'] = stream.get('codec_name')
                    break

            return info
        except Exception as e:
            logger.warning(f"无法获取视频信息: {e}")
            return {}

    def get_video_info(self, video_path: str) -> ProcessingResult:
        """获取视频信息（公开方法）

        Args:
            video_path: 视频文件路径

        Returns:
            ProcessingResult 处理结果，包含视频信息
        """
        try:
            info = self._get_video_info(video_path)
            if info:
                return ProcessingResult(
                    success=True,
                    output_path=video_path,
                    error_message=None,
                    duration=info.get('duration'),
                    metadata=info
                )
            else:
                return ProcessingResult(
                    success=False,
                    output_path=None,
                    error_message="无法获取视频信息"
                )
        except Exception as e:
            return ProcessingResult(
                success=False,
                output_path=None,
                error_message=str(e)
            )
    
    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        sample_rate: int = 44100,
        format: str = "wav",
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        从视频中提取音频
        
        Args:
            video_path: 输入视频路径
            output_path: 输出音频路径
            sample_rate: 采样率，默认 44100 Hz
            format: 输出格式，支持 "wav" 或 "mp3"
            progress_callback: 进度回调函数
        
        Returns:
            ProcessingResult 处理结果
        """
        logger.info(f"开始提取音频: {video_path} -> {output_path}")
        
        if not os.path.exists(video_path):
            return ProcessingResult(False, None, f"输入文件不存在: {video_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 根据格式设置编码器
        if format.lower() == "mp3":
            codec = "libmp3lame"
            bitrate = "192k"
        else:
            codec = "pcm_s16le"
            bitrate = None
        
        args = [
            "-i", video_path,
            "-vn",  # 禁用视频
            "-ar", str(sample_rate),  # 设置采样率
            "-ac", "2",  # 立体声
            "-c:a", codec,
        ]
        
        if bitrate:
            args.extend(["-b:a", bitrate])
        
        args.extend([
            "-y",  # 覆盖输出文件
            output_path
        ])
        
        success, error = self._run_ffmpeg(args, progress_callback)
        
        if success:
            duration = self._get_duration(output_path)
            logger.info(f"音频提取完成: {output_path}")
            return ProcessingResult(True, output_path, None, duration)
        else:
            return ProcessingResult(False, None, error)
    
    def cut_video(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: str,
        mode: str = 'fast',
        video_info: Optional[Dict[str, Any]] = None,  # 优化#4: 预缓存的视频信息
        progress_callback: Optional[Callable] = None,
        output_spec: Optional[OutputResolutionSpec] = None,
    ) -> ProcessingResult:
        """
        按时间范围切割视频
        
        Args:
            video_path: 输入视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出视频路径
            mode: 切割模式
                - 'fast': 流复制模式（-c copy），速度极快，但切割点可能偏移到最近关键帧
                - 'precise': 重新编码模式（h264_nvenc GPU加速），切割精确但较慢
            video_info: 预缓存的视频信息（避免重复调用 ffprobe），None 则自动获取
            progress_callback: 进度回调函数
        
        Returns:
            ProcessingResult 处理结果
        """
        output_path = os.path.splitext(output_path)[0] + ".mp4"
        
        # ── 视频切片安全保护：结束边界增加安全边距，防止包含下一段 ──
        SAFETY_MARGIN = 0.15
        safe_end_time = end_time - SAFETY_MARGIN
        safe_start_time = start_time + SAFETY_MARGIN
        
        # 确保调整后的时间依然有效
        if safe_end_time <= safe_start_time:
            # 如果太短了，用原始时间但至少留一点空间
            safe_end_time = end_time - 0.05
            safe_start_time = start_time
        
        logger.info(f"开始切割视频: {video_path} [{safe_start_time}s - {safe_end_time}s] (original=[{start_time}-{end_time}]) mode={mode} -> {output_path}")
        
        if not os.path.exists(video_path):
            return ProcessingResult(False, None, f"输入文件不存在: {video_path}")
        
        if safe_start_time >= safe_end_time:
            return ProcessingResult(False, None, "开始时间必须小于结束时间")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 统一导出规范（mode 保留仅用于兼容，不影响媒体参数）
        duration = safe_end_time - safe_start_time
        normalized_mode = (mode or "fast").lower()
        if normalized_mode not in {"fast", "accurate", "precise"}:
            normalized_mode = "fast"

        if output_spec is None:
            if not video_info:
                video_info = self._get_video_info(video_path)
            source_width = int((video_info or {}).get("width") or 0)
            source_height = int((video_info or {}).get("height") or 0)
            is_portrait = source_height > source_width if source_width and source_height else False
            target_w, target_h = (1080, 1920) if is_portrait else (1920, 1080)
        else:
            target_w = int(output_spec.width)
            target_h = int(output_spec.height)

        scale_filter = build_cover_crop_filter(target_w, target_h)

        common_args = [
            "-ss", str(safe_start_time),
            "-t", str(duration),
            "-i", video_path,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-vf", scale_filter,
            "-r", "60",
            *STANDARD_VIDEO_OUTPUT_ARGS,
            "-c:a", "aac",
            "-b:a", "160k",
            "-ac", "2",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-f", "mp4",
        ]

        # Use ASCII-only temp filename to avoid Chinese encoding issues in subprocess
        import tempfile as _tf
        _temp_dir = os.path.dirname(os.path.abspath(output_path))
        _temp_prefix = os.path.basename(output_path).split('.')[0] if '.' in os.path.basename(output_path) else 'clip'
        _temp_hash = hash(os.path.abspath(output_path)) % 100000
        temp_nvenc_output = os.path.join(_temp_dir, f"_nvenc_{_temp_hash:05x}.tmp.mp4")
        nvenc_args = common_args + [
            "-c:v", "h264_nvenc",
            "-rc", "cbr",
            "-b:v", "5M",
            "-maxrate", "5M",
            "-bufsize", "10M",
            "-preset", "p4",
            "-g", "120",
            "-y", temp_nvenc_output,
        ]

        success, error = self._run_ffmpeg(nvenc_args, progress_callback)
        if success:
            try:
                os.replace(temp_nvenc_output, output_path)
            except OSError as e:
                return ProcessingResult(False, None, f"写入输出文件失败: {e}")
            output_duration = self._get_duration(output_path)
            logger.info(f"视频切割完成: {output_path} (mode={normalized_mode}, encoder=nvenc)")
            return ProcessingResult(True, output_path, None, output_duration)

        if os.path.exists(temp_nvenc_output):
            try:
                os.remove(temp_nvenc_output)
            except OSError:
                pass

        logger.warning("NVENC 切片失败，回退 libx264: %s", error)
        # libx264 fallback: also use ASCII temp path
        temp_cpu_output = os.path.join(_temp_dir, f"_cpu_{_temp_hash:05x}.tmp.mp4")
        cpu_args = common_args + [
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-g", "120",
            "-y", temp_cpu_output,
        ]
        cpu_success, cpu_error = self._run_ffmpeg(cpu_args, progress_callback)
        if cpu_success:
            try:
                os.replace(temp_cpu_output, output_path)
            except OSError as e:
                return ProcessingResult(False, None, f"libx264 rename失败: {e}")
            output_duration = self._get_duration(output_path)
            logger.info(f"视频切割完成: {output_path} (mode={normalized_mode}, encoder=libx264)")
            return ProcessingResult(True, output_path, None, output_duration)

        combined_error = f"NVENC失败: {error}\\nlibx264失败: {cpu_error}"
        return ProcessingResult(False, None, combined_error)
        
        duration = end_time - start_time
        
        if mode == 'fast':
            # 快速模式：流复制，不重新编码
            # 切割点会自动对齐到最近的关键帧（±2-5秒偏移）
            temp_fast_output = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_fast_{_temp_hash:05x}.tmp.mp4")
            args = [
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", video_path,
                "-c", "copy",          # 流复制：直接复制音视频流，无重新编码
                "-avoid_negative_ts", "make_zero",
                "-y",
                temp_fast_output
            ]
        else:
            # 精确模式：NVENC 硬件编码（GPU加速，比 CPU 快 5~10 倍）
            # 获取视频信息（优先使用预缓存，避免重复 ffprobe）
            if video_info is None:
                video_info = self._get_video_info(video_path)
            width = video_info.get('width', 1920)
            height = video_info.get('height', 1080)
            
            # 确保最低 1080p
            min_height = 1080
            if height < min_height:
                scale_factor = min_height / height
                new_width = int(width * scale_factor)
                scale_filter = f"scale={new_width}:{min_height}"
            else:
                scale_filter = None
            
            args = [
                "-ss", str(start_time),  # 开始时间（输入侧搜索，快速定位）
                "-t", str(duration),     # 持续时间
                "-i", video_path,
                "-c:v", "h264_nvenc",   # NVIDIA 硬件编码（GPU 加速）
                "-rc", "vbr",           # 可变码率模式
                "-cq", "23",            # 质量（类似 CRF，值越小质量越高）
                "-preset", "p4",        # NVENC 预设（p1最快-p7最慢）
                "-b:v", "8M",           # 最大比特率上限
                "-c:a", "aac",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",
            ]
            
            # 添加缩放滤镜（如果需要）
            if scale_filter:
                args.extend(["-vf", scale_filter])
            
            # GOP 优化
            args.extend([
                "-g", "48",
                "-keyint_min", "48",
                "-sc_threshold", "0",
            ])
            
            args.extend(["-y", os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_acc_{_temp_hash:05x}.tmp.mp4")])
        
        success, error = self._run_ffmpeg(args, progress_callback)
        
        if success:
            # Rename from ASCII temp name to original Chinese filename
            actual_output = args[-1]  # last arg is the output file
            if actual_output != output_path and os.path.exists(actual_output):
                try:
                    os.replace(actual_output, output_path)
                except OSError as e:
                    return ProcessingResult(False, None, f"重命名失败: {e}")
            output_duration = self._get_duration(output_path)
            logger.info(f"视频切割完成: {output_path} (mode={mode})")
            return ProcessingResult(True, output_path, None, output_duration)
        else:
            return ProcessingResult(False, None, error)
    
    def enhance_audio(
        self,
        audio_path: str,
        output_path: str,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        音频增强处理
        
        功能:
        - 音量标准化 (loudnorm)
        - 降噪处理
        - 保持音质
        
        Args:
            audio_path: 输入音频路径
            output_path: 输出音频路径
            progress_callback: 进度回调函数
        
        Returns:
            ProcessingResult 处理结果
        """
        logger.info(f"开始音频增强: {audio_path} -> {output_path}")
        
        if not os.path.exists(audio_path):
            return ProcessingResult(False, None, f"输入文件不存在: {audio_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 音频滤镜链
        # loudnorm: 音量标准化
        # afftdn: FFT 降噪
        # highpass/lowpass: 去除极低频和高频噪音
        audio_filter = (
            "highpass=f=80,"  # 高通滤波，去除低频噪音
            "lowpass=f=15000,"  # 低通滤波，去除高频噪音
            "afftdn=nf=-25,"  # FFT 降噪
            "loudnorm=I=-16:TP=-1.5:LRA=11"  # 音量标准化 (EBU R128)
        )
        
        args = [
            "-i", audio_path,
            "-af", audio_filter,
            "-c:a", "pcm_s16le",  # 16-bit PCM
            "-ar", "44100",  # 采样率
            "-ac", "2",  # 立体声
            "-y",
            output_path
        ]
        
        success, error = self._run_ffmpeg(args, progress_callback)
        
        if success:
            duration = self._get_duration(output_path)
            logger.info(f"音频增强完成: {output_path}")
            return ProcessingResult(True, output_path, None, duration)
        else:
            return ProcessingResult(False, None, error)
    
    def transcode_video(
        self,
        video_path: str,
        output_path: str,
        min_resolution: int = 1080,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        视频转码
        
        确保输出不低于指定分辨率，保持高质量编码
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            min_resolution: 最小垂直分辨率（默认 1080）
            progress_callback: 进度回调函数
        
        Returns:
            ProcessingResult 处理结果
        """
        logger.info(f"开始视频转码: {video_path} -> {output_path}")
        
        if not os.path.exists(video_path):
            return ProcessingResult(False, None, f"输入文件不存在: {video_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 获取视频信息
        video_info = self._get_video_info(video_path)
        width = video_info.get('width', 1920)
        height = video_info.get('height', 1080)
        
        # 计算输出分辨率
        if height < min_resolution:
            scale_factor = min_resolution / height
            new_width = int(width * scale_factor)
            # 确保宽度是偶数（H.264 要求）
            new_width = new_width + (new_width % 2)
            new_height = min_resolution
            video_filter = f"scale={new_width}:{new_height}:flags=lanczos"
            logger.info(f"视频放大: {width}x{height} -> {new_width}x{new_height}")
        else:
            video_filter = "format=yuv420p"
            new_width, new_height = width, height
        
        args = [
            "-i", video_path,
            "-c:v", "h264_nvenc",   # NVIDIA 硬件编码
            "-rc", "vbr",
            "-cq", "18",            # 高质量
            "-preset", "p4",
            "-b:v", "15M",          # 高质量比特率上限
            "-c:a", "aac",
            "-b:a", "256k",  # 高质量音频
            "-ar", "48000",
            "-vf", video_filter,
            "-pix_fmt", "yuv420p",  # 兼容性
            "-movflags", "+faststart", # 网络优化
            "-y",
            output_path
        ]
        
        success, error = self._run_ffmpeg(args, progress_callback)
        
        if success:
            duration = self._get_duration(output_path)
            metadata = {
                'width': new_width,
                'height': new_height,
                'original_width': width,
                'original_height': height
            }
            logger.info(f"视频转码完成: {output_path}")
            return ProcessingResult(True, output_path, None, duration, metadata)
        else:
            return ProcessingResult(False, None, error)
    
    def process_segments(
        self,
        video_path: str,
        segments: List[Tuple[float, float, str]],
        output_dir: str,
        singer: str,
        song: str,
        enhance_audio: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """
        批量处理视频片段
        
        Args:
            video_path: 输入视频路径
            segments: 片段列表，格式 [(start_time, end_time, label), ...]
            output_dir: 输出目录
            singer: 歌手名称
            song: 歌曲名称
            enhance_audio: 是否增强音频
            progress_callback: 进度回调函数，接收 (current_segment, total_segments, progress_percent)
        
        Returns:
            List[ProcessingResult] 处理结果列表
        """
        logger.info(f"开始批量处理: {video_path}, 共 {len(segments)} 个片段")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"输入文件不存在: {video_path}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 清理文件名中的非法字符
        def sanitize_filename(name: str) -> str:
            """清理文件名中的非法字符"""
            # 替换 Windows 文件系统非法字符
            illegal_chars = '<>:"/\\|?*'
            for char in illegal_chars:
                name = name.replace(char, '_')
            return name.strip()
        
        singer_clean = sanitize_filename(singer)
        song_clean = sanitize_filename(song)
        
        results = []
        total_segments = len(segments)
        
        for idx, (start, end, label) in enumerate(segments, 1):
            logger.info(f"处理片段 {idx}/{total_segments}: {label} [{start}s - {end}s]")
            
            # 构建输出文件名: {歌手}_{歌曲}_{标签}_{序号}.mp4
            label_clean = sanitize_filename(label)
            output_filename = f"{singer_clean}_{song_clean}_{label_clean}_{idx:03d}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # 片段进度回调
            def segment_progress(progress, current_time, total_duration):
                if progress_callback:
                    overall_progress = ((idx - 1) + progress / 100) / total_segments * 100
                    progress_callback(idx, total_segments, overall_progress)
            
            # 切割视频
            result = self.cut_video(
                video_path=video_path,
                start_time=start,
                end_time=end,
                output_path=output_path,
                mode='precise',  # 优化#10: 修复参数名（原 quality='high' 会报 TypeError）
                progress_callback=segment_progress
            )
            
            # 如果需要音频增强且切割成功
            if enhance_audio and result.success:
                temp_audio = output_path + ".temp.wav"
                enhanced_audio = output_path + ".enhanced.wav"
                
                try:
                    # 提取音频
                    extract_result = self.extract_audio(
                        output_path, temp_audio, progress_callback=None
                    )
                    
                    if extract_result.success:
                        # 增强音频
                        enhance_result = self.enhance_audio(
                            temp_audio, enhanced_audio, progress_callback=None
                        )
                        
                        if enhance_result.success:
                            # 合并增强后的音频回视频
                            final_output = output_path + ".final.mp4"
                            merge_success = self._merge_audio_video(
                                output_path, enhanced_audio, final_output
                            )
                            
                            if merge_success:
                                os.replace(final_output, output_path)
                            
                            # 清理临时文件
                            if os.path.exists(enhanced_audio):
                                os.remove(enhanced_audio)
                        
                        if os.path.exists(temp_audio):
                            os.remove(temp_audio)
                
                except Exception as e:
                    logger.warning(f"音频增强失败: {e}")
            
            results.append(result)
            
            # 更新总体进度
            if progress_callback:
                progress_callback(idx, total_segments, (idx / total_segments) * 100)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"批量处理完成: {successful}/{total_segments} 成功")
        
        return results
    
    def burn_subtitle(
        self,
        video_path: str,
        output_path: str,
        label_text: str,
        font_path: str = r"C:\Windows\Fonts\msyh.ttc",
        font_size: int = 48,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """
        将分类标签文字烧录到视频左上角（硬字幕）

        Args:
            video_path:  输入视频路径
            output_path: 输出视频路径
            label_text:  要显示的文字，如 "副歌（高潮）"
            font_path:   字体文件路径（需支持中文）
            font_size:   字体大小（像素）
            progress_callback: 进度回调

        Returns:
            ProcessingResult
        """
        logger.info(f"烧录字幕: '{label_text}' -> {output_path}")

        if not os.path.exists(video_path):
            return ProcessingResult(False, None, f"输入文件不存在: {video_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Windows 路径分隔符需要转义给 FFmpeg drawtext
        font_path_escaped = font_path.replace('\\', '/').replace(':', '\\:')

        # drawtext 滤镜：左上角，半透明黑色背景，白色文字
        drawtext = (
            f"drawtext="
            f"fontfile='{font_path_escaped}':"
            f"text='{label_text}':"
            f"fontsize={font_size}:"
            f"fontcolor=white:"
            f"x=30:y=30:"
            f"box=1:"
            f"boxcolor=black@0.55:"
            f"boxborderw=10"
        )

        args = [
            "-i", video_path,
            "-vf", drawtext,
            "-c:v", "h264_nvenc",   # NVIDIA 硬件编码
            "-rc", "vbr",
            "-cq", "18",
            "-preset", "p4",
            "-c:a", "copy",
            "-y",
            output_path
        ]

        success, error = self._run_ffmpeg(args, progress_callback)

        if success:
            duration = self._get_duration(output_path)
            logger.info(f"字幕烧录完成: {output_path}")
            return ProcessingResult(True, output_path, None, duration)
        else:
            logger.warning(f"字幕烧录失败（跳过）: {error}")
            return ProcessingResult(False, None, error)

    def _merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> bool:
        """合并视频和音频"""
        args = [
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            output_path
        ]
        
        success, _ = self._run_ffmpeg(args, None)
        return success


def main():
    """示例用法"""
    # 创建处理器实例
    processor = FFmpegProcessor()
    
    # 示例: 提取音频
    # result = processor.extract_audio(
    #     video_path="input.mp4",
    #     output_path="output.wav"
    # )
    # print(f"提取结果: {result}")
    
    # 示例: 切割视频
    # result = processor.cut_video(
    #     video_path="input.mp4",
    #     start_time=10.5,
    #     end_time=30.0,
    #     output_path="output.mp4"
    # )
    # print(f"切割结果: {result}")
    
    # 示例: 批量处理
    # segments = [
    #     (0, 30, "intro"),
    #     (30, 60, "verse1"),
    #     (60, 90, "chorus"),
    # ]
    # results = processor.process_segments(
    #     video_path="input.mp4",
    #     segments=segments,
    #     output_dir="output",
    #     singer="周杰伦",
    #     song="晴天"
    # )
    pass


if __name__ == "__main__":
    main()
