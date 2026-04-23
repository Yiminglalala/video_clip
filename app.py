# -*- coding: utf-8 -*-
"""
视频字幕生成器 - Streamlit 主应用
======================================
使用豆包API进行语音识别
支持音频预处理（高通滤波+音量标准化）
自动生成 ASS 字幕并烧录到视频
"""

import os
import sys
import time
import shutil
import subprocess
import warnings
import tempfile
import json
import numpy as np
import scipy.signal as signal
import streamlit as st
from src.doubao_api import DoubaoASR, format_result
from src.output_spec import (
    DEFAULT_LANDSCAPE_RESOLUTION,
    LANDSCAPE_RESOLUTION_CHOICES,
    OutputResolutionSpec,
    SUBTITLE_FONT_FAMILY,
    build_cover_crop_filter,
    build_ass_filter_value,
    normalize_landscape_resolution_choice,
    resolve_output_resolution_spec,
    STANDARD_VIDEO_OUTPUT_ARGS,
)

warnings.filterwarnings("ignore")

# 项目根目录加入 path
PROJECT_ROOT = r"D:\video_clip"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
BACKUP_ROOT = os.path.join(PROJECT_ROOT, "backup_project")
if BACKUP_ROOT not in sys.path:
    sys.path.append(BACKUP_ROOT)

# 确保输出目录存在
os.makedirs(os.path.join(PROJECT_ROOT, "output"), exist_ok=True)


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="🎤 视频字幕生成器",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义 CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
.stApp { font-family: 'Noto Sans SC', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 16px;
    color: white; text-align: center; margin-bottom: 2rem;
}
.main-header h1 { margin: 0; font-size: 2.5rem; }
.main-header p { margin-top: 0.5rem; opacity: 0.9; font-size: 1.1rem; }
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 12px; padding: 1rem;
    border: 1px solid rgba(255,255,255,0.1); text-align: center;
}
.result-box {
    background: #f8f9fa; border-left: 4px solid #667eea;
    padding: 1rem; border-radius: 8px; margin: 1rem 0;
}
.success-box { border-color: #28a745; background: #f0fff0; }
.warning-box { border-color: #ffc107; background: #fffef0; }
.error-box   { border-color: #dc3545; background: #fff0f0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 全局状态管理
# ============================================================
def init_session_state():
    """初始化 session state"""
    defaults = {
        "preprocessing": True,             # 是否预处理
        "subtitle_mode": "sentence",       # word | sentence
        "output_dir": os.path.join(PROJECT_ROOT, "output"),
        "last_result": None,               # 上次处理结果
        "processing": False,
        "process_logs": [],
        # 豆包API配置
        "doubao_appid": "6118416182",
        "doubao_access_token": "wgYVCSXYek6ATuLNP_DiXFNHZ9jo5ZRV",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# 配置持久化管理
# ============================================================
CONFIG_FILE = os.path.join(PROJECT_ROOT, "slice_config.json")

DEFAULT_SLICE_CONFIG = {
    "min_dur": 10.0,
    "max_dur": 20.0,
    "cut_mode": "fast",
    "enable_subtitle": False,
    "singer_name": "",
    "concert_name": "",
    "landscape_resolution_choice": DEFAULT_LANDSCAPE_RESOLUTION,
}

def load_slice_config():
    """从文件加载切片配置"""
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_SLICE_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        # 合并默认值和保存值，确保兼容性
        config = DEFAULT_SLICE_CONFIG.copy()
        config.update(saved)
        return config
    except Exception as e:
        print(f"加载配置失败，使用默认值: {e}")
        return DEFAULT_SLICE_CONFIG.copy()

def save_slice_config(config):
    """保存切片配置到文件"""
    try:
        # 只保存可序列化的配置项
        to_save = {k: v for k, v in config.items() 
                   if k in DEFAULT_SLICE_CONFIG.keys()}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"配置已保存: {CONFIG_FILE}")
    except Exception as e:
        print(f"保存配置失败: {e}")


RESOLUTION_LABELS = {
    "1920x1080": "1920x1080（横屏）",
    "1080x1440": "1080x1440（3:4）",
    "1080x1920": "1080x1920（竖屏）",
}


def get_landscape_resolution_choice() -> str:
    config = st.session_state.get("slice_config") or load_slice_config()
    stored = config.get("landscape_resolution_choice", DEFAULT_LANDSCAPE_RESOLUTION)
    return normalize_landscape_resolution_choice(stored)


def get_output_resolution_spec(video_path: str, landscape_choice: str | None = None) -> OutputResolutionSpec:
    orientation = detect_orientation(video_path)
    choice = normalize_landscape_resolution_choice(
        landscape_choice or get_landscape_resolution_choice()
    )
    return resolve_output_resolution_spec(orientation, choice)


init_session_state()


# ============================================================
# 音频预处理函数
# ============================================================
def preprocess_audio(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """
    音频预处理流水线（演唱会人声增强版）：
    1. 转单声道
    2. 带通滤波 100-4000Hz（人声频段，去除鼓/贝斯/镲片/高频噪音）
    3. 预加重（增强辅音清晰度，ASR 关键优化）
    4. 动态压缩（压缩音量动态范围，让轻声也够响）
    5. 峰值归一化 peak → 0.9
    
    针对演唱会场景：乐器多、观众噪音大、人声忽大忽小
    """
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Step 2: 带通滤波 100-4000Hz（保留人声基频+谐波，切掉低频鼓/高频噪音）
    sos_bp = signal.butter(4, [100.0, 4000.0], btype='band', fs=sr, output='sos')
    filtered = signal.sosfilt(sos_bp, audio_data)
    
    # Step 3: 预加重（一阶高通，系数0.97，提升辅音/t/s/f/sh/等高频细节）
    filtered = np.append(filtered[0], filtered[1:] - 0.97 * filtered[:-1])
    
    # Step 4: 动态压缩（soft-clip: 把 >阈值的信号压缩到线性范围）
    threshold = 0.15 * np.max(np.abs(filtered))
    if threshold > 0:
        abs_sig = np.abs(filtered)
        # 超过阈值的部分用 tanh 压缩（平滑，不产生硬削波）
        mask = abs_sig > threshold
        compressed = filtered.copy()
        compressed[mask] = (
            np.sign(filtered[mask]) *
            (threshold + (abs_sig[mask] - threshold) * 
             np.tanh((abs_sig[mask] - threshold) / (threshold + 1e-8)))
        )
        filtered = compressed
    
    # Step 5: 峰值归一化到 0.9
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        normalized = filtered * (0.9 / max_val)
    else:
        normalized = filtered
    
    return normalized.astype(np.float32)


def extract_audio_from_video(video_path: str, sr: int = 16000) -> tuple:
    """
    从视频中提取音频
    Returns: (audio_array, sample_rate)
    """
    import librosa
    import soundfile as sf

    direct_error = None
    try:
        audio, orig_sr = librosa.load(video_path, sr=sr, mono=True)
        if not orig_sr or float(orig_sr) <= 0:
            raise ValueError(f"librosa 返回了无效采样率: {orig_sr}")
        if audio is None or np.size(audio) == 0:
            raise ValueError("librosa 读取到空音频")
        return np.asarray(audio, dtype=np.float32), int(orig_sr)
    except ZeroDivisionError as e:
        direct_error = e
    except Exception as e:
        direct_error = e

    # 回退：用 ffmpeg 强制提取 16k mono wav，规避部分容器/音轨导致的 sr=0 问题
    temp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav = tmp.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sr)),
            "-f",
            "wav",
            temp_wav,
        ]
        run = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120,
        )
        if run.returncode != 0:
            err_tail = (run.stderr or "").strip()[-400:]
            raise RuntimeError(f"ffmpeg 提取失败: {err_tail or '未知错误'}")

        audio, out_sr = sf.read(temp_wav, dtype="float32", always_2d=False)
        if audio is None or np.size(audio) == 0:
            raise RuntimeError("ffmpeg 提取后音频为空")
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if not out_sr or float(out_sr) <= 0:
            raise RuntimeError(f"ffmpeg 提取结果采样率无效: {out_sr}")
        if int(out_sr) != int(sr):
            audio = librosa.resample(audio, orig_sr=int(out_sr), target_sr=int(sr))
            out_sr = int(sr)
        return np.asarray(audio, dtype=np.float32), int(out_sr)
    except subprocess.TimeoutExpired:
        raise RuntimeError("音频提取超时（>120s），请检查视频是否损坏。")
    except Exception as fallback_error:
        raise RuntimeError(
            f"音频提取失败：direct={direct_error}; fallback={fallback_error}"
        )
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass


# ============================================================
# ASR 引擎封装
# ============================================================

def run_asr(audio: np.ndarray, sr: int, do_preprocess: bool = True) -> dict:
    """
    统一 ASR 接口
    
    Args:
        audio: 音频 numpy 数组
        sr: 采样率
        do_preprocess: 是否做预处理
    
    Returns:
        统一格式的结果字典
    """
    start_total = time.time()
    
    # 预处理：增强人声
    if do_preprocess:
        proc_audio = preprocess_audio(audio.copy(), sr)
    else:
        proc_audio = audio
    
    # 豆包API处理
    appid = st.session_state.get("doubao_appid", "")
    access_token = st.session_state.get("doubao_access_token", "")
    if not appid or not access_token:
        return {
            "engine": "doubao",
            "text": "",
            "words": [],
            "sentences": [],
            "language": "zh",
            "error": "豆包API配置未完成，请在高级设置中填写appid和access_token"
        }
    
    # 将音频转换为WAV格式
    import soundfile as sf
    import io
    buffer = io.BytesIO()
    sf.write(buffer, proc_audio, sr, format="wav")
    audio_data = buffer.getvalue()
    
    try:
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
    
    result["total_time"] = round(time.time() - start_total, 2)
    result["preprocessed"] = do_preprocess
    result["duration_sec"] = round(len(audio) / sr, 1)
    
    return result


# ============================================================
# 字幕生成 & 视频烧录
# ============================================================

def _auto_wrap_text(text, max_chars=14):
    """自动换行：在标点符号或适当位置换行"""
    if len(text) <= max_chars:
        return text
    
    lines = []
    current = ""
    punctuations = "。！？、，；：,.;:!?"
    i = 0
    n = len(text)
    
    while i < n:
        remaining = text[i:]
        if len(remaining) <= max_chars:
            lines.append(remaining)
            break
        
        # 找最佳换行位置
        break_pos = -1
        # 优先在标点符号后换行
        for j in range(min(max_chars, i + max_chars // 2, i + max_chars)):
            if j < n and text[j] in punctuations:
                break_pos = j + 1
                break
        
        # 如果没找到标点，尝试在逗号或顿号后换行
        if break_pos == -1:
            for j in range(min(max_chars, i + max_chars // 2, i + max_chars)):
                if j < n and text[j] in "，、, ":
                    break_pos = j + 1
                    break
        
        # 如果还是没找到，强制在 max_chars 位置换行
        if break_pos == -1:
            break_pos = i + max_chars
        
        lines.append(text[i:break_pos])
        i = break_pos
    
    return "\\N".join(lines)


def _find_split_index(text: str, max_chars: int = 14) -> int:
    if len(text) <= max_chars:
        return len(text)

    preferred_start = max(1, max_chars - 4)
    preferred_end = min(len(text) - 1, max_chars + 2)
    punctuations = "。！？、，；：,.;:!? "

    for idx in range(preferred_end, preferred_start - 1, -1):
        if text[idx - 1] in punctuations:
            return idx

    return min(max_chars, len(text) - 1)


def _split_long_sentence_entry(sent: dict, max_chars: int = 14) -> list[dict]:
    text = str(sent.get("text", "") or "").strip()
    if not text or len(text) <= max_chars:
        return [sent]

    split_idx = _find_split_index(text, max_chars=max_chars)
    first_text = text[:split_idx].strip()
    second_text = text[split_idx:].strip()
    if not first_text or not second_text:
        return [sent]

    start = float(sent.get("start", 0.0) or 0.0)
    end = float(sent.get("end", start + 3.0) or (start + 3.0))
    total_duration = max(end - start, 0.0)

    if total_duration <= 0.12:
        split_time = start + 0.06
    else:
        ratio = len(first_text) / max(len(first_text) + len(second_text), 1)
        split_time = start + total_duration * ratio
        min_gap = min(0.18, total_duration / 3.0)
        split_time = max(start + min_gap, min(end - min_gap, split_time))

    first_sent = dict(sent)
    first_sent["text"] = first_text
    first_sent["start"] = start
    first_sent["end"] = split_time

    second_sent = dict(sent)
    second_sent["text"] = second_text
    second_sent["start"] = split_time
    second_sent["end"] = end

    return [first_sent, second_sent]


def generate_ass_from_sentences(sentences, ass_path, output_spec: OutputResolutionSpec | None = None, orientation="landscape"):
    """
    从句子列表生成 ASS 字幕文件
    """
    events = []
    expanded_sentences = []
    for sent in sentences:
        expanded_sentences.extend(_split_long_sentence_entry(sent, max_chars=14))
    for i, sent in enumerate(expanded_sentences):
        start = sent.get("start", 0)
        end = sent.get("end", start + 3)
        text = sent.get("text", "")
        
        # 格式化时间
        def format_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h}:{m:02d}:{s:05.2f}"
        
        start_str = format_time(start)
        end_str = format_time(end)
        
        # 转义 ASS 特殊字符
        text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N").replace("\r", "")
        
        # 自动换行处理
        
        events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0000,0000,0000,,{text}")
    
    if output_spec is None:
        output_spec = resolve_output_resolution_spec(orientation, DEFAULT_LANDSCAPE_RESOLUTION)
    play_res_x = output_spec.width
    play_res_y = output_spec.height
    font_size = 90
    
    # ASS 文件内容
    events_text = "\n".join(events)
    ass_content = f"""
[Script Info]
ScriptType: v4.00+
Title: Subtitles
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{SUBTITLE_FONT_FAMILY},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,100,100,240,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{events_text}
"""
    
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ass_content)


def generate_ass_from_words(words, ass_path, output_spec: OutputResolutionSpec | None = None, orientation="landscape"):
    """
    从单词列表生成 ASS 字幕文件
    """
    events = []
    for i, word in enumerate(words):
        start = word.get("start", 0)
        end = word.get("end", start + 1)
        text = word.get("text", word.get("word", ""))
        
        # 格式化时间
        def format_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h}:{m:02d}:{s:05.2f}"
        
        start_str = format_time(start)
        end_str = format_time(end)
        
        # 转义 ASS 特殊字符
        text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N").replace("\r", "")
        
        events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0000,0000,0000,,{text}")
    
    if output_spec is None:
        output_spec = resolve_output_resolution_spec(orientation, DEFAULT_LANDSCAPE_RESOLUTION)

    # ASS 文件内容
    events_text = "\n".join(events)
    ass_content = f"""
[Script Info]
ScriptType: v4.00+
Title: Subtitles
PlayResX: {output_spec.width}
PlayResY: {output_spec.height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{SUBTITLE_FONT_FAMILY},90,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,100,100,240,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{events_text}
"""
    
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ass_content)


def words_to_sentence_chunks(words, max_chars=24, max_gap=0.6):
    """Fallback: merge word-level tokens into sentence-level subtitle chunks."""
    if not words:
        return []

    chunks = []
    current_text = ""
    current_start = None
    current_end = None
    last_end = None

    def flush_chunk():
        nonlocal current_text, current_start, current_end
        text = (current_text or "").strip()
        if text and current_start is not None and current_end is not None:
            chunks.append({"text": text, "start": current_start, "end": current_end})
        current_text = ""
        current_start = None
        current_end = None

    for item in words:
        text = (item.get("text") or item.get("word") or "").strip()
        if not text:
            continue

        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        gap = 0.0 if last_end is None else max(0.0, start - last_end)

        if current_start is None:
            current_start = start
            current_end = end
            current_text = text
            last_end = end
            continue

        need_break = (
            gap > max_gap
            or len(current_text) + len(text) > max_chars
            or (current_text and current_text[-1] in "。！？!?,，、；;：:")
        )
        if need_break:
            flush_chunk()
            current_start = start
            current_end = end
            current_text = text
        else:
            if (
                current_text
                and current_text[-1].isascii()
                and current_text[-1].isalnum()
                and text[0].isascii()
                and text[0].isalnum()
            ):
                current_text += " "
            current_text += text
            current_end = end

        last_end = end

    flush_chunk()
    return chunks


def burn_subtitle_from_file(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    output_spec: OutputResolutionSpec | None = None,
    orientation: str = "landscape",
) -> tuple:
    """
    使用现有的字幕文件烧录到视频
    
    Args:
        video_path: 输入视频路径
        subtitle_path: 字幕文件路径（.ass格式）
        output_path: 输出视频路径
        orientation: 视频方向 ("portrait" 或 "landscape")
        
    Returns:
        (success, output_path_or_error)
    """
    import subprocess
    import shutil
    
    # 检查字幕文件是否存在
    if not os.path.exists(subtitle_path):
        return False, f"字幕文件不存在: {subtitle_path}"
    
    # 检查字幕文件格式
    if not subtitle_path.endswith(".ass"):
        return False, "只支持 .ass 格式的字幕文件"
    
    # FFmpeg 烧录字幕（Windows安全：cwd+相对路径，避免冒号问题）
    video_dir = os.path.dirname(video_path)
    ass_local = os.path.join(video_dir, os.path.basename(subtitle_path))
    same_ass_path = os.path.abspath(ass_local) == os.path.abspath(subtitle_path)
    if not same_ass_path:
        shutil.copy2(subtitle_path, ass_local)
    
    if output_spec is None:
        output_spec = resolve_output_resolution_spec(orientation, DEFAULT_LANDSCAPE_RESOLUTION)
    target_w = output_spec.width
    target_h = output_spec.height
    target_dar = f"{target_w}/{target_h}"
    ass_filter = build_ass_filter_value(os.path.basename(ass_local), post_filter=f"setsar=1,setdar={target_dar}")
    video_filter = build_cover_crop_filter(target_w, target_h, extra_filter=ass_filter)

    nvenc_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", video_filter,
        "-c:v", "h264_nvenc",
        "-rc", "cbr", "-b:v", "8M", "-maxrate", "8M", "-bufsize", "16M", "-preset", "p4",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        *STANDARD_VIDEO_OUTPUT_ARGS,
        "-r", "60",
        "-movflags", "+faststart",
        output_path,
    ]
    cpu_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", video_filter,
        "-c:v", "libx264",
        "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        *STANDARD_VIDEO_OUTPUT_ARGS,
        "-r", "60",
        "-movflags", "+faststart",
        output_path,
    ]
    
    try:
        r = subprocess.run(nvenc_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=300, cwd=video_dir)
        if r.returncode != 0 and r.stderr and "h264_nvenc" in r.stderr:
            r = subprocess.run(cpu_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=300, cwd=video_dir)
        if r.returncode != 0:
            error_msg = r.stderr[-500:] if r.stderr else "未知错误"
            return False, f"FFmpeg 错误:\n{error_msg}"
        return True, output_path
    except subprocess.TimeoutExpired:
        return False, "FFmpeg 编码超时（>5分钟）"
    except Exception as e:
        return False, str(e)
    finally:
        # 清理视频目录中的临时ASS
        if os.path.exists(ass_local) and (os.path.abspath(ass_local) != os.path.abspath(subtitle_path)):
            os.remove(ass_local)


def generate_video_with_subtitles(
    video_path: str,
    asr_result: dict,
    output_path: str,
    output_spec: OutputResolutionSpec | None = None,
    orientation: str = "landscape",
    subtitle_mode: str = "sentence",
) -> tuple:
    """
    完整流程：ASR结果 → ASS字幕 → FFmpeg烧录
    
    Returns: (success, output_path_or_error)
    """
    import subprocess
    
    temp_dir = st.session_state.output_dir
    os.makedirs(temp_dir, exist_ok=True)
    fd, ass_path = tempfile.mkstemp(suffix=".ass", dir=temp_dir, prefix="temp_subtitle_")
    os.close(fd)
    if output_spec is None:
        output_spec = resolve_output_resolution_spec(orientation, DEFAULT_LANDSCAPE_RESOLUTION)
    
    # 选择字幕模式
    if asr_result.get("sentences"):
        generate_ass_from_sentences(asr_result["sentences"], ass_path, output_spec=output_spec, orientation=orientation)
    elif asr_result.get("words"):
        fallback_sentences = words_to_sentence_chunks(asr_result["words"])
        if fallback_sentences:
            generate_ass_from_sentences(fallback_sentences, ass_path, output_spec=output_spec, orientation=orientation)
        else:
            return False, "No usable ASR text"
    else:
        return False, "没有可用的识别文本"
    
    # 调用通用的字幕烧录函数
    result = burn_subtitle_from_file(video_path, ass_path, output_path, output_spec=output_spec, orientation=orientation)
    
    # 清理临时ASS文件
    if os.path.exists(ass_path):
        os.remove(ass_path)
    
    return result


def detect_orientation(video_path: str) -> str:
    """
    检测视频方向（考虑旋转元数据）
    """
    import subprocess
    import json

    def _to_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default

    def _get_rotation_deg(stream: dict) -> int:
        # 1) tags.rotate
        tags = stream.get("tags", {}) if isinstance(stream.get("tags", {}), dict) else {}
        rotate_tag = tags.get("rotate")
        if rotate_tag is not None:
            return _to_int(rotate_tag, 0)

        # 2) side_data_list.rotation / displaymatrix rotation
        for item in stream.get("side_data_list", []) or []:
            if not isinstance(item, dict):
                continue
            if "rotation" in item:
                return _to_int(item.get("rotation"), 0)
            if item.get("side_data_type") == "Display Matrix":
                # 某些 ffprobe 版本会在 displaymatrix 文字里带 "rotation of -90.00 degrees"
                txt = str(item.get("displaymatrix", ""))
                marker = "rotation of"
                idx = txt.lower().find(marker)
                if idx >= 0:
                    try:
                        tail = txt[idx + len(marker):].strip().split()[0]
                        return _to_int(tail, 0)
                    except Exception:
                        pass
        return 0

    try:
        # 使用ffprobe获取视频信息
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            return "landscape"

        info = json.loads(result.stdout)
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                width = _to_int(stream.get('width', 0), 0)
                height = _to_int(stream.get('height', 0), 0)
                if width <= 0 or height <= 0:
                    continue

                rotation = _get_rotation_deg(stream) % 360
                # 旋转 90/270 时，显示宽高交换
                if rotation in (90, 270):
                    width, height = height, width

                if width > height:
                    return "landscape"
                else:
                    return "portrait"
    except Exception:
        pass
    
    # 默认返回横屏
    return "landscape"


def _load_json_safe(payload: str):
    """Parse ffprobe json output safely."""
    if payload is None:
        return {}
    text = str(payload).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def render_video_info(video_path: str):
    """显示视频信息卡片（稳健版，避免 ffprobe JSON 异常）。"""
    if not video_path or not os.path.exists(video_path):
        st.warning(f"无法读取视频信息: 文件不存在 - {video_path}")
        return

    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json", "-show_format", "-show_streams",
        video_path,
    ]

    try:
        r = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
        if r.returncode != 0:
            detail = (r.stderr or "").strip()[-300:] if r.stderr else "ffprobe 执行失败"
            st.warning(f"无法读取视频信息: {detail}")
            st.caption(f"路径: `{video_path}`")
            return

        info = _load_json_safe(r.stdout)
        if not isinstance(info, dict):
            st.warning("无法读取视频信息: ffprobe 返回了空/非 JSON 内容")
            st.caption(f"路径: `{video_path}`")
            return

        fmt = info.get("format", {}) if isinstance(info.get("format", {}), dict) else {}
        try:
            size_mb = float(fmt.get("size", 0) or 0) / (1024 * 1024)
        except Exception:
            size_mb = 0.0
        try:
            duration = float(fmt.get("duration", 0) or 0)
        except Exception:
            duration = 0.0

        video_stream = None
        audio_stream = None
        for s in info.get("streams", []):
            if s.get("codec_type") == "video" and not video_stream:
                video_stream = s
            elif s.get("codec_type") == "audio" and not audio_stream:
                audio_stream = s

        cols = st.columns(5)
        with cols[0]:
            st.metric("时长", f"{duration:.1f}s")
        with cols[1]:
            if video_stream:
                width = int(float(video_stream.get("width", 0) or 0))
                height = int(float(video_stream.get("height", 0) or 0))
                st.metric("分辨率", f"{width}x{height}" if width > 0 and height > 0 else "N/A")
            else:
                st.metric("分辨率", "N/A")
        with cols[2]:
            st.metric("大小", f"{size_mb:.1f} MB")
        with cols[3]:
            orient = detect_orientation(video_path)
            orient_text = "竖屏" if orient == "portrait" else ("横屏" if orient == "landscape" else "自动")
            st.metric("方向", orient_text)
        with cols[4]:
            st.metric("音频流", "有" if audio_stream is not None else "无")

        st.caption(f"路径: `{video_path}`")
    except Exception as e:
        st.warning(f"无法读取视频信息: {e}")
        st.caption(f"路径: `{video_path}`")


def _segment_label_to_display_type(label: str) -> str:
    label_norm = (label or "").strip().lower()
    mapping = {
        "verse": "主歌",
        "chorus": "副歌",
        "audience": "合唱",
        "crowd": "合唱",
        "speech": "讲话串场",
        "talk": "讲话串场",
        "solo": "乐器SOLO",
    }
    return mapping.get(label_norm, "未标注")


def _infer_type_from_filename(name: str) -> str:
    stem = os.path.splitext(name)[0]
    if "讲话串场" in stem:
        return "讲话串场"
    if "乐器SOLO" in stem or "独奏" in stem:
        return "乐器SOLO"
    if "主歌" in stem:
        return "主歌"
    if "副歌" in stem:
        return "副歌"
    if "合唱" in stem:
        return "合唱"

    lower_stem = stem.lower()
    if "_verse_" in lower_stem or lower_stem.endswith("_verse"):
        return "主歌"
    if "_chorus_" in lower_stem or lower_stem.endswith("_chorus"):
        return "副歌"
    if "_audience_" in lower_stem or "_crowd_" in lower_stem:
        return "合唱"
    if "_speech_" in lower_stem or "_talk_" in lower_stem:
        return "讲话串场"
    if "_solo_" in lower_stem:
        return "乐器SOLO"
    return "未标注"


def _is_highlight_meta(seg_obj, filename: str) -> bool:
    if "高光" in filename or "highlight" in filename.lower():
        return True
    features = getattr(seg_obj, "features", None)
    if not features:
        return False
    for key in ("is_highlight", "highlight", "high_light"):
        if hasattr(features, "get"):
            val = features.get(key)
        else:
            val = getattr(features, key, None)
        if bool(val):
            return True
    return False


def render_slicing_mode():
    """视频智能切片模式 - 四步工作流：选择视频 -> 处理 -> 预览编辑 -> 导出"""
    try:
        from src.processor import LiveVideoProcessor, ProcessingConfig
        from src.audio_analyzer import LABEL_CN, LABEL_COLORS, LABEL_INTRO, LABEL_VERSE, LABEL_CHORUS, LABEL_OUTRO, LABEL_INTERLUDE, LABEL_SOLO, LABEL_TALK, LABEL_SPEECH, LABEL_CROWD, LABEL_AUDIENCE, LABEL_OTHER, LABEL_SILENCE
    except Exception as e:
        st.error("切片模块未就绪：无法导入 `src.processor` 或 `src.audio_analyzer`。")
        st.code(str(e))
        return

    # 初始化工作流 session state
    if 'slice_workflow_step' not in st.session_state:
        st.session_state.slice_workflow_step = 0
    if 'slice_video_source' not in st.session_state:
        st.session_state.slice_video_source = None
    if 'slice_analysis_result' not in st.session_state:
        st.session_state.slice_analysis_result = None
    if 'slice_output_files' not in st.session_state:
        st.session_state.slice_output_files = None
    if 'slice_export_segments' not in st.session_state:
        st.session_state.slice_export_segments = None
    if 'slice_segments' not in st.session_state:
        st.session_state.slice_segments = []
    if 'slice_selected_segment' not in st.session_state:
        st.session_state.slice_selected_segment = None
    if 'slice_jump_to_time' not in st.session_state:
        st.session_state.slice_jump_to_time = None
    if 'slice_auto_play_segment' not in st.session_state:
        st.session_state.slice_auto_play_segment = None
    if 'slice_processing_logs' not in st.session_state:
        st.session_state.slice_processing_logs = []

    # 工作流步骤定义（内部用 0-3，UI 显示 1-4）
    WORKFLOW_STEPS = {
        0: {"name": "选择视频", "icon": "📹", "desc": "上传或选择本地视频文件"},
        1: {"name": "结构分析", "icon": "⚙️", "desc": "执行音频结构分析，识别段落类型"},
        2: {"name": "预览编辑", "icon": "✏️", "desc": "查看并编辑片段，点击时间轴跳转"},
        3: {"name": "导出视频", "icon": "🎬", "desc": "导出选中的视频片段"},
    }

    # 显示工作流进度条
    st.header("🎬 演唱会视频智能切片 v3.0")
    cols = st.columns(4)
    current_step = st.session_state.slice_workflow_step
    for i, col in enumerate(cols):
        step = WORKFLOW_STEPS[i]
        is_completed = i < current_step
        is_current = i == current_step
        with col:
            if is_completed:
                st.markdown(f"### ✅ {step['icon']} {step['name']}")
            elif is_current:
                st.markdown(f"### 🔵 {step['icon']} {step['name']}")
            else:
                st.markdown(f"### ⚪ {step['icon']} {step['name']}")
            st.caption(step['desc'])

    st.markdown("---")

    # 根据当前步骤渲染对应的 UI
    if current_step == 0:
        _render_step0_select_video()
    elif current_step == 1:
        _render_step1_processing()
    elif current_step == 2:
        _render_step2_preview_edit()
    elif current_step == 3:
        _render_step3_export()

# ============================================================
# 四步工作流步骤函数
# ============================================================

def _render_step0_select_video():
    """步骤 1：选择视频文件与切片参数（无识曲配置）"""
    st.subheader("📤 步骤 1：选择视频文件")
    
    # 加载保存的配置
    saved_config = load_slice_config()

    uploaded_file = st.file_uploader(
        "上传演唱会视频",
        help="输入格式不限（可解码即可），最终导出固定为 MP4",
        key="slice_step0_uploader",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        local_path = st.text_input("或粘贴本地路径", value="", key="slice_step0_local_path")
    with col2:
        use_local = st.button("使用本地文件", use_container_width=True, key="slice_step0_use_local_btn")

    video_source = None
    if local_path.strip() and os.path.exists(local_path.strip()):
        video_source = local_path.strip()
    elif use_local and local_path.strip():
        st.warning(f"文件不存在: {local_path.strip()}")
    elif uploaded_file is not None:
        temp_dir = os.path.join(st.session_state.output_dir, "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        upload_ext = os.path.splitext(getattr(uploaded_file, "name", "") or "")[1].lower() or ".bin"
        video_source = os.path.join(temp_dir, f"upload_{int(time.time())}{upload_ext}")
        with open(video_source, "wb") as f:
            f.write(uploaded_file.getvalue())

    if not video_source:
        st.info("请上传视频或粘贴本地路径")
        return

    st.session_state.slice_video_source = video_source
    render_video_info(video_source)
    input_orientation = detect_orientation(video_source)
    landscape_resolution_choice = normalize_landscape_resolution_choice(
        saved_config.get("landscape_resolution_choice", DEFAULT_LANDSCAPE_RESOLUTION)
    )

    info_col1, info_col2 = st.columns([2, 3])
    with info_col1:
        singer_name = st.text_input(
            "🎤 歌手名称", 
            value=saved_config.get("singer_name", ""),
            key="slice_step0_singer_name"
        )
    with info_col2:
        concert_name = st.text_input(
            "🎵 演唱会名称", 
            value=saved_config.get("concert_name", ""), 
            placeholder="可选", 
            key="slice_step0_concert_name"
        )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        min_dur = st.number_input(
            "最小时长（秒）",
            value=float(saved_config.get("min_dur", 10.0)),
            min_value=8.0,
            max_value=30.0,
            step=1.0,
            key="slice_step0_min_dur",
        )
    with c2:
        max_dur = st.number_input(
            "最大时长（秒）",
            value=float(saved_config.get("max_dur", 20.0)),
            min_value=8.0,
            max_value=30.0,
            step=1.0,
            key="slice_step0_max_dur",
        )
    with c3:
        cut_mode = st.selectbox(
            "切片模式",
            ["fast", "accurate"],
            index=0 if saved_config.get("cut_mode", "fast") == "fast" else 1,
            format_func=lambda x: "⚡ 快速（关键帧）" if x == "fast" else "🎯 精确（逐帧）",
            key="slice_step0_cut_mode",
        )
    with c4:
        enable_subtitle = st.checkbox(
            "📝 生成字幕", 
            value=saved_config.get("enable_subtitle", False), 
            key="slice_step0_enable_subtitle"
        )

    if input_orientation == "portrait":
        st.info("输出分辨率：竖屏视频按项目原规范输出 1080x1920")
    else:
        landscape_resolution_choice = st.selectbox(
            "输出分辨率",
            list(LANDSCAPE_RESOLUTION_CHOICES),
            index=list(LANDSCAPE_RESOLUTION_CHOICES).index(landscape_resolution_choice),
            format_func=lambda x: RESOLUTION_LABELS.get(x, x),
            key="slice_step0_landscape_resolution_choice",
        )

    st.caption('时长规则：8 ≤ 最小时长 ≤ 最大时长 ≤ 30（默认 10-20）｜勾选"生成字幕"将用豆包API识别歌词并烧录字幕')

    # 更新并保存配置
    current_config = {
        "min_dur": min_dur,
        "max_dur": max_dur,
        "cut_mode": cut_mode,
        "enable_subtitle": enable_subtitle,
        "singer_name": singer_name,
        "concert_name": concert_name,
        "landscape_resolution_choice": landscape_resolution_choice,
    }
    st.session_state.slice_config = current_config
    
    # 自动保存配置
    save_slice_config(current_config)

    duration_valid = 8.0 <= float(min_dur) <= float(max_dur) <= 30.0
    if not duration_valid:
        st.error("时长参数不合法：必须满足 8 <= 最小时长 <= 最大时长 <= 30")
        return

    st.markdown("---")
    if st.button("➡️ 下一步：开始处理", type="primary", use_container_width=True, key="slice_step0_next_btn"):
        st.session_state.slice_workflow_step = 1
        st.rerun()


def _render_step1_processing():
    """步骤 2: 处理视频"""
    from src.processor import LiveVideoProcessor, ProcessingConfig
    from src.audio_analyzer import LABEL_CN, LABEL_COLORS, LABEL_INTRO, LABEL_VERSE, LABEL_CHORUS, LABEL_OUTRO, LABEL_INTERLUDE, LABEL_SOLO, LABEL_TALK, LABEL_SPEECH, LABEL_CROWD, LABEL_AUDIENCE, LABEL_OTHER, LABEL_SILENCE
    
    st.subheader("⚙️ 步骤 2: 视频结构分析")

    video_source = st.session_state.slice_video_source
    if not video_source or not os.path.exists(video_source):
        st.error("❌ 视频文件不存在，请重新选择")
        if st.button("⬅️ 返回选择视频", key="slice_step1_back_btn"):
            st.session_state.slice_workflow_step = 0
            st.rerun()
        return

    st.info(f"📂 处理视频: {os.path.basename(video_source)}")

    # 显示处理流程说明
    st.markdown("#### 将要执行的流程：")
    cols = st.columns(4)
    cols[0].markdown("1. 🔍 **音频分析**\n\n识别音乐片段类型")
    cols[1].markdown("2. 📝 **智能分类**\n\n主歌/副歌/合唱/讲话等")
    cols[2].markdown("3. 🎵 **场景分割**\n\n按内容自动切分片段")
    cols[3].markdown("4. 🏷️ **智能标注**\n\n为每个片段添加标签")

    st.markdown("---")

    # 检查是否已完成处理（rerun 后保留状态）
    processing_done = bool(st.session_state.slice_segments)

    if not processing_done:
        # 未处理：显示开始处理按钮
        if st.button("🚀 开始分析", type="primary", use_container_width=True, key="slice_step1_start_btn"):
            # 检查 CUDA
            songformer_device = "cuda"
            try:
                import torch
                cuda_ok = bool(torch.cuda.is_available())
                if cuda_ok:
                    gpu_name = torch.cuda.get_device_name(0)
                    props = torch.cuda.get_device_properties(0)
                    gpu_mem_bytes = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
                    gpu_mem = gpu_mem_bytes / (1024**3)
                    st.info(f"✅ GPU 可用：{gpu_name} ({gpu_mem:.1f}GB)，所有模型将使用 CUDA 加速")
                    songformer_device = "cuda"
                else:
                    st.warning("⚠️ **CUDA 不可用**：SongFormer / Demucs / FireRed 将以 CPU 运行，速度会显著下降。建议检查 GPU 驱动。")
                    songformer_device = "cpu"
            except Exception as torch_err:
                st.warning(f"⚠️ 无法检测 CUDA 状态：{torch_err}，将尝试 auto 模式。")
                songformer_device = "auto"

            # 获取配置
            config_dict = st.session_state.slice_config
            config = ProcessingConfig(
                output_dir=st.session_state.output_dir,
                min_segment_duration=float(config_dict.get("min_dur", 8.0)),
                max_segment_duration=float(config_dict.get("max_dur", 15.0)),
                min_duration_limit=float(config_dict.get("min_dur", 8.0)),
                max_duration_limit=float(config_dict.get("max_dur", 15.0)),
                enable_songformer=True,
                strict_songformer=True,
                songformer_device=songformer_device,
                songformer_window=60,
                songformer_hop=30,
                concert=config_dict.get("concert_name") or None,
                cut_mode=config_dict.get("cut_mode", "fast"),
                enable_subtitle=config_dict.get("enable_subtitle", False),
                subtitle_mode="sentence",
            )

            # 检查 SongFormer 依赖
            if config.enable_songformer and config.strict_songformer:
                try:
                    from src.songformer_analyzer import SongFormerAnalyzer
                    ok, missing = SongFormerAnalyzer.check_runtime_dependencies()
                except Exception as dep_err:
                    st.error(f"SongFormer 依赖预检失败: {dep_err}")
                    return
                if not ok:
                    st.error(f"SongFormer 缺少依赖，请先安装后再切片：\n\n`python -m pip install {' '.join(missing)}`")
                    return

            # 显示进度条
            st.session_state.slice_processing_logs = []
            progress_bar = st.progress(0, "准备中...")
            status_text = st.empty()
            log_box = st.empty()

            def on_progress(pct, msg):
                progress_bar.progress(min(pct, 1.0), msg)
                status_text.info(msg)
                ts = time.strftime("%H:%M:%S")
                st.session_state.slice_processing_logs.append(f"[{ts}] {msg}")
                log_box.markdown("#### 分析日志\n" + "\n".join(f"- {line}" for line in st.session_state.slice_processing_logs[-80:]))

            try:
                on_progress(0.01, "初始化处理器...")
                processor = LiveVideoProcessor(config)
                on_progress(0.02, "开始执行结构分析...")

                # 只做分析，不导出视频
                result, export_segments, subtitled_video_path = processor.analyze_video(
                    video_source,
                    singer=config_dict.get("singer_name") or None,
                    concert=config_dict.get("concert_name") or None,
                    progress_callback=on_progress,
                )

                # 保存分析结果
                st.session_state.slice_analysis_result = result
                st.session_state.slice_export_segments = export_segments
                st.session_state.slice_output_files = []  # 导出阶段才有
                # 保存烧录后的视频路径（如果有）
                st.session_state.slice_subtitled_video_path = subtitled_video_path
                # 保存ASR缓存
                st.session_state.slice_cached_asr_results = getattr(processor, '_cached_asr_results', {})
                # 保存字幕启用状态
                st.session_state.slice_enable_subtitle = config_dict.get("enable_subtitle", False)
                
                # 保存歌曲信息（供编辑用）
                songs_info = []
                for song in result.songs:
                    songs_info.append({
                        "song_index": getattr(song, 'song_index', len(songs_info)),
                        "song_title": getattr(song, 'song_title', f"Song_{len(songs_info)+1:02d}"),
                        "song_artist": getattr(song, 'song_artist', ''),
                        "start_time": float(getattr(song, 'start_time', 0.0)),
                        "end_time": float(getattr(song, 'end_time', 0.0)),
                    })
                st.session_state.slice_songs_info = songs_info

                # 从最终导出片段构建前端数据（已是合并/拆分后的）
                flat_segments = []
                for idx, item in enumerate(export_segments):
                    export_type = item["type"]
                    flat_segments.append({
                        "id": f"seg_{idx:03d}",
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "songformer_label": item.get("songformer_label", ""),  # SongFormer 英文原始标签
                        "current_label": export_type,  # 映射后的输出标签
                        "_initial_label": export_type,  # 初始输出标签（用于重置）
                        "is_highlight": bool(item.get("is_highlight", False)),
                        "confidence": getattr(item.get("segment"), "confidence", 0.9),
                        "modified": False,
                        "song_index": getattr(item.get("segment"), "song_index", 0),
                        # 保留原始切片点（用于显示对齐信息）
                        "original_start": float(item.get("original_start", item["start"])),
                        "original_end": float(item.get("original_end", item["end"])),
                    })

                st.session_state.slice_segments = flat_segments

                progress_bar.progress(1.0, "分析完成!")
                st.success(f"✅ 分析完成！共 **{len(result.songs)}** 首歌，**{len(flat_segments)}** 个最终片段")
                st.session_state.slice_workflow_step = 2
                st.rerun()

            except Exception as e:
                import traceback
                st.error(f"❌ 分析失败:\n{e}\n\n```\n{traceback.format_exc()[-2000:]}\n```")
    else:
        # 已分析完成：显示结果摘要 + 下一步按钮
        segments = st.session_state.slice_segments
        analysis = st.session_state.slice_analysis_result
        song_count = len(analysis.songs) if analysis else 0
        st.success(f"✅ 分析完成！共 **{song_count}** 首歌，**{len(segments)}** 个最终片段")

        # 显示片段摘要（输出标签 + SongFormer 标签 + 时长）
        summary_data = []
        for idx, seg in enumerate(segments[:10]):
            label = seg['current_label']
            sf_label = seg.get('songformer_label', '')
            dur = seg['end'] - seg['start']
            sf_part = f" ({sf_label})" if sf_label else ""
            summary_data.append(f"`{idx+1}` {label}{sf_part} ({dur:.1f}s)")
        st.markdown("  •  ".join(summary_data))
        if len(segments) > 10:
            st.caption(f"...还有 {len(segments) - 10} 个片段")

        if st.button("➡️ 下一步：预览与编辑", type="primary", use_container_width=True, key="slice_step1_next_btn"):
            st.session_state.slice_workflow_step = 2
            st.rerun()

    # 返回按钮
    st.markdown("---")
    if st.button("⬅️ 返回选择视频", key="slice_step1_back_btn"):
        st.session_state.slice_workflow_step = 0
        st.rerun()


def _rebuild_flat_segments_from_export_segments(export_segments):
    """从导出片段恢复前端编辑用的扁平片段列表。"""
    flat_segments = []
    for idx, item in enumerate(export_segments or []):
        export_type = item.get("type", "主歌")
        flat_segments.append({
            "id": f"seg_{idx:03d}",
            "start": float(item.get("start", 0.0)),
            "end": float(item.get("end", 0.0)),
            "songformer_label": item.get("songformer_label", ""),
            "current_label": export_type,
            "_initial_label": export_type,
            "is_highlight": bool(item.get("is_highlight", False)),
            "confidence": getattr(item.get("segment"), "confidence", 0.9),
            "modified": False,
            "song_index": getattr(item.get("segment"), "song_index", 0),
            "original_start": float(item.get("original_start", item.get("start", 0.0))),
            "original_end": float(item.get("original_end", item.get("end", 0.0))),
        })
    return flat_segments


def _render_step2_preview_edit():
    """步骤 3: 预览与编辑"""
    from src.audio_analyzer import LABEL_CN, LABEL_COLORS, LABEL_INTRO, LABEL_VERSE, LABEL_CHORUS, LABEL_OUTRO, LABEL_INTERLUDE, LABEL_SOLO, LABEL_TALK, LABEL_SPEECH, LABEL_CROWD, LABEL_AUDIENCE, LABEL_OTHER, LABEL_SILENCE

    st.subheader("✏️ 步骤 3: 视频预览与片段编辑（点击时间轴跳转到对应时间）")

    video_source = st.session_state.slice_video_source
    segments = st.session_state.slice_segments or []
    if not segments:
        export_segments = st.session_state.get("slice_export_segments") or []
        if export_segments:
            segments = _rebuild_flat_segments_from_export_segments(export_segments)
            st.session_state.slice_segments = segments

    if not segments:
        st.warning("⚠️ 未找到片段数据，请重新处理视频")
        if st.button("⬅️ 返回重新处理", key="slice_step2_back_btn"):
            st.session_state.slice_workflow_step = 1
            st.rerun()
        return

    processing_logs = st.session_state.get("slice_processing_logs", [])
    if processing_logs:
        with st.expander("查看本次分析日志", expanded=False):
            st.markdown("\n".join(f"- {line}" for line in processing_logs[-120:]))

    # 1. 视频预览（选中片段时动态生成该片段的预览）
    jump_time = st.session_state.slice_jump_to_time
    st.markdown("**📺 视频预览**")

    # 获取当前选中片段
    selected_idx = st.session_state.slice_selected_segment
    video_to_use = video_source
    use_dynamic_preview = False

    if selected_idx is not None and 0 <= selected_idx < len(segments):
        seg = segments[selected_idx]
        use_dynamic_preview = True

        # 检查缓存的预览视频是否是当前片段
        cached_preview_key = f"preview_{selected_idx}_{seg['start']:.2f}_{seg['end']:.2f}"
        cached_preview_path = st.session_state.get(f"slice_preview_{selected_idx}_path")
        
        # 检查缓存是否有效
        if cached_preview_path and os.path.exists(cached_preview_path):
            # 验证缓存是否对应当前片段
            last_preview_key = st.session_state.get(f"slice_preview_{selected_idx}_key")
            if last_preview_key == cached_preview_key:
                video_to_use = cached_preview_path
                st.info(f"✅ 使用已缓存的片段 #{selected_idx+1} 预览")
            else:
                # 清除旧缓存
                try:
                    if os.path.exists(cached_preview_path):
                        os.remove(cached_preview_path)
                except:
                    pass
                cached_preview_path = None
        
        # 没有缓存或缓存过期，生成新预览
        if not cached_preview_path:
            # 检查是否有ASR缓存
            cached_asr = st.session_state.get('slice_cached_asr_results', {})
            songs = getattr(st.session_state.get('slice_analysis_result'), 'songs', [])
            song_index = seg.get('song_index', 0)
            
            try:
                # 生成预览片段
                temp_dir = st.session_state.output_dir
                os.makedirs(temp_dir, exist_ok=True)
                
                # 1. 先切原始视频（不带字幕）
                from src.ffmpeg_processor import FFmpegProcessor
                from src.output_spec import build_cover_crop_filter, build_ass_filter_value, resolve_output_resolution_spec, normalize_landscape_resolution_choice, DEFAULT_LANDSCAPE_RESOLUTION, OutputResolutionSpec
                
                ffmpeg = FFmpegProcessor()
                temp_no_subtitle = os.path.join(temp_dir, f"temp_preview_nosub_{selected_idx}.mp4")
                
                output_spec = get_output_resolution_spec(video_source)
                
                # 安全边距：和最终导出一致
                safety_margin = 0.15
                safe_start = seg['start'] + safety_margin
                safe_end = seg['end'] - safety_margin
                if safe_end <= safe_start:
                    safe_end = seg['end'] - 0.05
                    safe_start = seg['start']
                
                cut_result = ffmpeg.cut_video(
                    video_source,
                    safe_start,
                    safe_end,
                    temp_no_subtitle,
                    mode="accurate",
                    output_spec=output_spec
                )
                
                if cut_result.success:
                    # 2. 如果有字幕，再烧录
                    temp_final = temp_no_subtitle
                    if cached_asr and st.session_state.get('slice_enable_subtitle', False):
                        # 找到该歌曲的ASR缓存
                        cached = cached_asr.get(song_index)
                        if cached and not cached.get('error'):
                            # 找到该歌曲的start_time
                            song_start_time = 0.0
                            for song in songs:
                                if getattr(song, 'song_index', -1) == song_index:
                                    song_start_time = float(getattr(song, 'start_time', 0.0))
                                    break
                            
                            # 计算相对于歌曲的时间
                            relative_start = safe_start - song_start_time
                            relative_end = safe_end - song_start_time
                            
                            # 调用processor的函数生成带字幕的视频
                            from src.processor import LiveVideoProcessor, ProcessingConfig
                            temp_subtitled = os.path.join(temp_dir, f"temp_preview_sub_{selected_idx}.mp4")
                            config = ProcessingConfig(
                                output_dir=temp_dir,
                                enable_subtitle=True,
                            )
                            dummy_processor = LiveVideoProcessor(config)
                            # 直接复用之前写的 _generate_subtitles_from_cached_asr
                            result_ok, result_path = dummy_processor._generate_subtitles_from_cached_asr(
                                temp_no_subtitle,
                                cached,
                                relative_start,
                                relative_end,
                                temp_subtitled,
                                output_spec
                            )
                            if result_ok:
                                temp_final = result_path
                    
                    # 3. 保存缓存
                    video_to_use = temp_final
                    st.session_state[f"slice_preview_{selected_idx}_path"] = temp_final
                    st.session_state[f"slice_preview_{selected_idx}_key"] = cached_preview_key
                    st.info(f"✅ 已生成片段 #{selected_idx+1} 的预览")
            except Exception as e:
                import traceback
                st.warning(f"⚠️ 生成切片预览失败：{e}")
                use_dynamic_preview = False
                video_to_use = None
    # 调整列比例，让视频区域更小
    col_video, col_info = st.columns([2, 1])

    with col_video:
        if video_to_use and os.path.exists(video_to_use):
            # 直接用 width 参数控制视频大小（更可靠）
            st.video(video_to_use, width=320)
            
            if use_dynamic_preview:
                # 清除自动播放标记
                st.session_state.slice_auto_play_segment = None
            else:
                # 原始视频播放逻辑（保持原样）
                if jump_time is not None:
                    st.session_state.slice_jump_to_time = None

                # 自动播放+自动停止：点击片段后 JS 控制 video 元素
                auto_play_idx = st.session_state.slice_auto_play_segment
                if auto_play_idx is not None:
                    segments = st.session_state.slice_segments
                    if 0 <= auto_play_idx < len(segments):
                        seg = segments[auto_play_idx]
                        start_t = seg['start']
                        end_t = seg['end']
                        js_code = f"""
                        <script>
                        (function() {{
                            var startT = {start_t};
                            var endT = {end_t};
                            function bindVideo() {{
                                var videos = window.parent.document.querySelectorAll('video');
                                if (!videos.length) {{ setTimeout(bindVideo, 200); return; }}
                                var v = videos[videos.length - 1];
                                v.currentTime = startT;
                                v.play().catch(function() {{}});
                                // 移除旧监听
                                if (v._sfStopHandler) v.removeEventListener('timeupdate', v._sfStopHandler);
                                v._sfStopHandler = function() {{
                                    if (v.currentTime >= endT) {{
                                        v.pause();
                                        v.removeEventListener('timeupdate', v._sfStopHandler);
                                        v._sfStopHandler = null;
                                    }}
                                }};
                                v.addEventListener('timeupdate', v._sfStopHandler);
                            }}
                            setTimeout(bindVideo, 300);
                        }})();
                        </script>
                        """
                        components.html(js_code, height=0)
                        st.session_state.slice_auto_play_segment = None
        elif selected_idx is None:
            st.info("请选择一个切片后再预览。当前页面只预览切片，不再显示整条视频。")
        else:
            st.warning("⚠️ 当前切片预览不可用，请重新选择该切片或重新处理。")

    with col_info:
        st.markdown("**📋 视频信息**")
        if video_source:
            video_name = os.path.basename(video_source)
            st.info(f"📁 **文件名**: {video_name}")

        total_duration = max(s['end'] for s in segments) if segments else 0.0
        st.metric("⏱️ 总时长", f"{total_duration:.1f}s")

        segment_count = len(segments)
        modified_count = sum(1 for s in segments if s.get('modified', False))
        st.metric("📊 片段数", f"{segment_count} (已修改: {modified_count})")

        st.markdown("---")
        st.caption("💡 **时间轴使用说明**")
        st.caption("- 点击下方任意时间轴片段")
        st.caption("- 自动跳转到该片段起始时间")
        st.caption("- 自动播放一次该片段")
        st.caption("- 可在展开面板中编辑标签")

    st.markdown("---")

    # 3. 歌曲信息编辑区
    st.markdown("#### 🎵 歌曲信息（可手动修改）")
    songs_info = st.session_state.get('slice_songs_info', [])
    if songs_info:
        for song_idx, song_info in enumerate(songs_info):
            with st.expander(f"🎵 第 {song_idx+1} 首歌 ({song_info['start_time']:.1f}s - {song_info['end_time']:.1f}s)", expanded=(song_idx == 0)):
                col_song_name, col_artist = st.columns([2, 1])
                with col_song_name:
                    new_title = st.text_input(
                        f"🎤 歌曲名称 {song_idx+1}",
                        value=song_info.get('song_title', f"Song_{song_idx+1:02d}"),
                        key=f"song_title_{song_idx}",
                    )
                with col_artist:
                    new_artist = st.text_input(
                        f"👤 歌手/艺术家 {song_idx+1}",
                        value=song_info.get('song_artist', ''),
                        key=f"song_artist_{song_idx}",
                    )
                
                # 更新到 session_state
                if new_title != song_info.get('song_title') or new_artist != song_info.get('song_artist'):
                    st.session_state.slice_songs_info[song_idx]['song_title'] = new_title
                    st.session_state.slice_songs_info[song_idx]['song_artist'] = new_artist
    else:
        st.info("没有识别到歌曲信息")

    st.markdown("---")

    # 4. 片段时间轴（可点击跳转）
    st.markdown("#### ⏱️ 片段时间轴（点击跳转到对应时间）")

    total_duration = max(s['end'] for s in segments) if segments else 0.0
    st.info(f"📌 总时长: {total_duration:.1f}s | 🔢 共 {len(segments)} 个片段")

    # 分组显示切片按钮，每组最多 8 个，保持横向布局
    max_cols_per_row = 8
    num_rows = (len(segments) + max_cols_per_row - 1) // max_cols_per_row
    
    for row in range(num_rows):
        start_idx = row * max_cols_per_row
        end_idx = min(start_idx + max_cols_per_row, len(segments))
        row_segments = segments[start_idx:end_idx]
        
        timeline_cols = st.columns(len(row_segments))
        
        for local_idx, (col, seg) in enumerate(zip(timeline_cols, row_segments)):
            global_idx = start_idx + local_idx
            label = seg['current_label']
            sf_label = seg.get('songformer_label', '')
            color = LABEL_COLORS.get(label, '#9E9E9E')
            duration = seg['end'] - seg['start']

            with col:
                is_modified = seg.get('modified', False)
                is_selected = st.session_state.slice_selected_segment == global_idx
                state_part = "\n[已选中]" if is_selected else ("\n[已修改]" if is_modified else "")
                sf_part = f"\nSF:{sf_label}" if sf_label else ""
                button_label = f"{label}{state_part}\n{duration:.1f}s{sf_part}\n[{seg['start']:.1f}-{seg['end']:.1f}]"
                button_type = "primary" if is_selected else "secondary"

                if st.button(button_label, type=button_type, key=f"timeline_jump_btn_{global_idx}"):
                    st.session_state.slice_jump_to_time = seg['start']
                    st.session_state.slice_selected_segment = global_idx
                    st.session_state.slice_auto_play_segment = global_idx
                    st.rerun()

    # 时间刻度
    st.markdown("##### 时间刻度")
    num_ticks = min(5, int(total_duration / 10) + 1)
    tick_cols = st.columns(num_ticks)
    for i, col in enumerate(tick_cols):
        tick_time = (total_duration / num_ticks) * i
        col.caption(f"{tick_time:.0f}s")

    # 图例
    st.markdown("##### 图例")
    used_labels = list(set(s['current_label'] for s in segments))
    legend_cols = st.columns(min(6, len(used_labels)))
    for col, label in zip(legend_cols, used_labels[:6]):
        color = LABEL_COLORS.get(label, '#9E9E9E')
        col.markdown(
            f"<span style=\"display:inline-block;width:14px;height:14px;"
            f"background:{color};border-radius:3px;margin-right:6px;"
            f"vertical-align:middle;\"></span>{label}",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 3. 片段编辑面板
    if st.session_state.slice_selected_segment is not None:
        idx = st.session_state.slice_selected_segment
        if 0 <= idx < len(segments):
            _render_segment_editor(idx)

    st.markdown("---")

    # 4. 操作按钮
    col_back, col_reset, col_next = st.columns([1, 1, 2])

    with col_back:
        if st.button("⬅️ 返回重新处理", key="slice_step2_back_btn"):
            st.session_state.slice_workflow_step = 1
            st.session_state.slice_selected_segment = None
            st.rerun()

    with col_reset:
        if st.button("🔄 重置所有修改", key="slice_step2_reset_btn"):
            for seg in st.session_state.slice_segments:
                seg['current_label'] = seg.get('_initial_label', seg['current_label'])
                seg['modified'] = False
            st.success("✅ 已重置所有修改")
            st.rerun()

    with col_next:
        if st.button("➡️ 下一步：导出视频", type="primary", key="slice_step2_next_btn"):
            if not st.session_state.slice_segments:
                export_segments = st.session_state.get("slice_export_segments") or []
                if export_segments:
                    st.session_state.slice_segments = _rebuild_flat_segments_from_export_segments(export_segments)
            st.session_state.slice_workflow_step = 3
            st.rerun()


def _render_segment_editor(idx):
    """渲染片段编辑器"""
    from src.audio_analyzer import LABEL_CN, LABEL_COLORS

    segments = st.session_state.slice_segments
    seg = segments[idx]

    with st.expander(f"🎯 编辑片段 #{idx + 1}", expanded=True):
        label = seg['current_label']
        sf_label = seg.get('songformer_label', '')
        duration = seg['end'] - seg['start']

        # 显示字幕对齐信息
        if 'original_start' in seg and 'original_end' in seg:
            orig_start = seg['original_start']
            orig_end = seg['original_end']
            shift_start = seg['start'] - orig_start
            shift_end = seg['end'] - orig_end
            if abs(shift_start) > 0.01 or abs(shift_end) > 0.01:
                st.success(
                    f"✅ 字幕对齐: [原: {orig_start:.1f}-{orig_end:.1f}] "
                    f"→ [现: {seg['start']:.1f}-{seg['end']:.1f}] "
                    f"(偏移: {shift_start:+.1f}s / {shift_end:+.1f}s)"
                )

        # 双标签展示
        sf_part = f" | **SongFormer**: {sf_label}" if sf_label else ""
        st.info(f"**输出标签**: {label}{sf_part} | **时长**: {duration:.1f}s")
        
        # 显示该片段内的字幕
        if st.session_state.get('slice_enable_subtitle', False):
            cached_asr = st.session_state.get('slice_cached_asr_results', {})
            songs = getattr(st.session_state.get('slice_analysis_result'), 'songs', [])
            song_index = seg.get('song_index', 0)
            
            try:
                from src.subtitle_alignment import get_segment_subtitles
                song_start_time = 0.0
                for song in songs:
                    if getattr(song, 'song_index', -1) == song_index:
                        song_start_time = float(getattr(song, 'start_time', 0.0))
                        break
                
                subtitles = get_segment_subtitles(
                    seg['start'], seg['end'],
                    song_index,
                    cached_asr,
                    song_start_time
                )
                
                if subtitles:
                    st.markdown("**📜 片段内的字幕**:")
                    for sub in subtitles:
                        text = sub.get('text', '')
                        if sub.get('is_truncated_start') or sub.get('is_truncated_end'):
                            st.warning(f"⚠️ [可能被切断] {text}")
                        else:
                            st.markdown(f"- {text}")
            except Exception:
                pass

        # 输出标签选项（最终导出用的类型）
        export_type_options = ["主歌", "副歌", "合唱", "讲话串场", "乐器SOLO"]
        current_idx = export_type_options.index(label) if label in export_type_options else 0

        new_label = st.selectbox(
            "输出标签", export_type_options,
            index=current_idx, key=f"slice_label_edit_{idx}"
        )

        # SongFormer 原始标签编辑
        sf_label_options = list(LABEL_CN.keys())
        sf_display_options = [f"{k} ({LABEL_CN[k]})" for k in sf_label_options]
        sf_current_idx = sf_label_options.index(sf_label) if sf_label in sf_label_options else 0

        new_sf_label = st.selectbox(
            "SongFormer 标签", sf_display_options,
            index=sf_current_idx, key=f"slice_sf_label_edit_{idx}"
        )
        # 提取英文 key
        new_sf_label = sf_label_options[sf_display_options.index(new_sf_label)]

        st.markdown("---")

        col_start, col_end = st.columns(2)
        with col_start:
            new_start = st.number_input(
                "开始时间 (秒)", value=float(seg['start']), min_value=0.0,
                max_value=float(seg['end']) - 0.1, step=0.1, key=f"slice_start_edit_{idx}"
            )
        with col_end:
            new_end = st.number_input(
                "结束时间 (秒)", value=float(seg['end']), min_value=float(new_start) + 0.1,
                step=0.1, key=f"slice_end_edit_{idx}"
            )

        st.markdown("---")

        col_apply, col_cancel = st.columns(2)
        with col_apply:
            if st.button("✅ 应用修改", type="primary", key=f"slice_apply_{idx}"):
                _apply_segment_edit(idx, new_label, float(new_start), float(new_end), new_sf_label)
                st.success("✅ 修改已应用！")
                st.session_state.slice_selected_segment = None
                st.rerun()
        with col_cancel:
            if st.button("❌ 取消编辑", key=f"slice_cancel_{idx}"):
                st.session_state.slice_selected_segment = None
                st.info("已取消编辑")
                st.rerun()


def _apply_segment_edit(idx, new_label, new_start, new_end, new_sf_label=None):
    """应用片段编辑"""
    segments = st.session_state.slice_segments
    seg = segments[idx]

    has_changes = (
        new_label != seg['current_label'] or
        abs(new_start - seg['start']) > 0.01 or
        abs(new_end - seg['end']) > 0.01 or
        (new_sf_label is not None and new_sf_label != seg.get('songformer_label', ''))
    )

    if has_changes:
        seg['current_label'] = new_label
        seg['start'] = new_start
        seg['end'] = new_end
        if new_sf_label is not None:
            seg['songformer_label'] = new_sf_label
        seg['modified'] = True

        # 调整相邻片段
        if idx > 0:
            prev_seg = segments[idx - 1]
            if abs(new_start - prev_seg['end']) > 0.01:
                prev_seg['end'] = new_start
                prev_seg['modified'] = True

        if idx < len(segments) - 1:
            next_seg = segments[idx + 1]
            if abs(new_end - next_seg['start']) > 0.01:
                next_seg['start'] = new_end
                next_seg['modified'] = True


def _render_step3_export():
    """步骤 4: 导出视频"""
    from src.processor import LiveVideoProcessor, ProcessingConfig

    st.subheader("🎬 步骤 3: 导出切片视频")

    segments = st.session_state.slice_segments or []
    export_segments = st.session_state.slice_export_segments or []
    if not segments and export_segments:
        segments = _rebuild_flat_segments_from_export_segments(export_segments)
        st.session_state.slice_segments = segments

    if not segments:
        st.warning("⚠️ 没有可以导出的视频片段，请重新处理视频。")
        if st.button("⬅️ 返回重新处理", key="slice_step3_back_btn"):
            st.session_state.slice_workflow_step = 1
            st.rerun()
        return

    total_segments = len(segments)
    modified_count = sum(1 for s in segments if s.get('modified', False))
    total_duration = max(s['end'] for s in segments) if segments else 0.0

    cols = st.columns(4)
    cols[0].metric("📊 总片段数", str(total_segments))
    cols[1].metric("✏️ 已修改", str(modified_count))
    cols[2].metric("🔄 修改率", f"{(modified_count / total_segments * 100):.0f}%" if total_segments else "0%")
    cols[3].metric("⏱️ 总时长", f"{total_duration:.1f}s")

    st.markdown("---")

    st.markdown("#### 📝 最终片段详情")
    table_data = []
    for idx, seg in enumerate(segments):
        label = seg['current_label']
        sf_label = seg.get('songformer_label', '')
        duration = seg['end'] - seg['start']
        modified = "是" if seg.get('modified', False) else "否"
        sf_part = f" ({sf_label})" if sf_label else ""
        table_data.append({
            "序号": idx + 1,
            "输出标签": label,
            "SongFormer": sf_label or "-",
            "开始时间": f"{seg['start']:.1f}s",
            "结束时间": f"{seg['end']:.1f}s",
            "时长": f"{duration:.1f}s",
            "已修改": modified,
        })
    st.dataframe(table_data, use_container_width=True)

    st.markdown("---")

    col_export, col_restart = st.columns([2, 1])
    with col_export:
        if st.button("🎬 导出切片视频", type="primary", key="slice_export_btn"):
            with st.spinner("正在导出视频..."):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                # 同步用户在 Step2 的编辑到 export_segments
                songs_info = st.session_state.get('slice_songs_info', [])
                for i, seg in enumerate(segments):
                    if i < len(export_segments):
                        export_segments[i]["start"] = seg['start']
                        export_segments[i]["end"] = seg['end']
                        export_segments[i]["type"] = seg['current_label']
                        export_segments[i]["songformer_label"] = seg.get('songformer_label', '')
                        
                        # 更新歌曲名和歌手（如果用户修改了）
                        seg_song_idx = seg.get('song_index', 0)
                        for song_info in songs_info:
                            if song_info.get('song_index', -1) == seg_song_idx:
                                song_obj = export_segments[i].get('song')
                                if song_obj:
                                    song_obj.song_title = song_info.get('song_title', song_obj.song_title)
                                    song_obj.song_artist = song_info.get('song_artist', song_obj.song_artist)

                # 获取配置（复用 Step1 的配置）
                config_dict = st.session_state.slice_config
                config = ProcessingConfig(
                    output_dir=st.session_state.output_dir,
                    min_segment_duration=float(config_dict.get("min_dur", 8.0)),
                    max_segment_duration=float(config_dict.get("max_dur", 15.0)),
                    min_duration_limit=float(config_dict.get("min_dur", 8.0)),
                    max_duration_limit=float(config_dict.get("max_dur", 15.0)),
                    enable_songformer=True,
                    strict_songformer=True,
                    songformer_device="cuda",
                    concert=config_dict.get("concert_name") or None,
                    cut_mode=config_dict.get("cut_mode", "fast"),
                    enable_subtitle=config_dict.get("enable_subtitle", False),
                    subtitle_mode="sentence",
                    landscape_resolution_choice=normalize_landscape_resolution_choice(
                        config_dict.get("landscape_resolution_choice", DEFAULT_LANDSCAPE_RESOLUTION)
                    ),
                    source_orientation=detect_orientation(st.session_state.slice_video_source),
                )

                def on_export_progress(pct, msg):
                    progress_bar.progress(min(pct, 1.0))
                    status_text.info(msg)

                try:
                    processor = LiveVideoProcessor(config)
                    # 优先使用烧录后的视频
                    subtitled_video_path = st.session_state.get('slice_subtitled_video_path')
                    video_to_export = st.session_state.slice_video_source
                    if subtitled_video_path and os.path.exists(subtitled_video_path):
                        video_to_export = subtitled_video_path
                        st.info("✅ 正在从已烧录字幕的视频切片")

                    output_files = processor.export_video_segments(
                        video_to_export,
                        export_segments,
                        progress_callback=on_export_progress,
                        singer=config_dict.get("singer_name") or None,
                    )

                    st.session_state.slice_output_files = output_files

                    progress_bar.progress(1.0)
                    st.success(f"✅ 成功导出 {len(output_files)} 个视频切片！")
                    if output_files:
                        output_dir = os.path.dirname(output_files[0])
                        st.info(f"📂 输出目录: {output_dir}")

                        # 预览第一个切片视频
                        preview_path = output_files[0]
                        if os.path.exists(preview_path):
                            st.markdown("**📹 切片预览：**")
                            st.video(preview_path)

                except Exception as e:
                    import traceback
                    st.error(f"❌ 导出失败:\n{e}\n\n```\n{traceback.format_exc()[-2000:]}\n```")

    with col_restart:
        if st.button("🔄 重新开始", key="slice_restart_btn"):
            # 重置所有工作流状态
            st.session_state.slice_workflow_step = 0
            st.session_state.slice_video_source = None
            st.session_state.slice_analysis_result = None
            st.session_state.slice_output_files = None
            st.session_state.slice_export_segments = None
            st.session_state.slice_segments = []
            st.session_state.slice_selected_segment = None
            st.session_state.slice_jump_to_time = None
            st.rerun()


# ============================================================
# 原有辅助函数（保留）
# ============================================================


# ============================================================
# UI 渲染
# ============================================================

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🎬 演唱会工具台</h1>
        <p>视频切片 + 字幕生成（简化版）</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """
    侧边栏：精简参数
    """
    with st.sidebar:
        st.markdown("### ⚙️ 设置")
        
        # 音频预处理
        st.session_state.preprocessing = st.checkbox(
            "音频预处理（人声增强）", 
            value=True,
            help="对音频进行滤波和音量标准化，提高识别准确率"
        )
        
        # 字幕模式固定为按句显示，避免逐字字幕影响观感
        st.session_state.subtitle_mode = "sentence"
        st.info("字幕模式已固定为：按句子显示（横排）")
        
        st.divider()
        
        # 豆包API配置
        st.markdown("### 📡 豆包API配置")
        st.session_state.doubao_appid = st.text_input(
            "AppID",
            value=st.session_state.doubao_appid,
            help="豆包API的AppID"
        )
        st.session_state.doubao_access_token = st.text_input(
            "Access Token",
            value=st.session_state.doubao_access_token,
            type="password",
            help="豆包API的Access Token"
        )


def render_subtitle_mode():
    """简化字幕页（豆包 API + 按句显示）。"""
    temp_dir = st.session_state.output_dir
    os.makedirs(temp_dir, exist_ok=True)

    st.markdown("### 📹 上传视频")
    uploaded_file = st.file_uploader(
        "选择要处理的视频文件",
        help="支持常见视频格式",
        key="subtitle_video_uploader",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        local_path = st.text_input(
            "或粘贴本地路径",
            value="",
            key="subtitle_local_path",
        )
    with col2:
        use_local = st.button(
            "使用本地文件",
            use_container_width=True,
            key="subtitle_use_local_file_btn",
        )

    video_path = None
    if uploaded_file:
        # 保存上传的视频
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"视频上传成功: {uploaded_file.name}")
    elif local_path.strip() and os.path.exists(local_path.strip()):
        video_path = local_path.strip()
        st.success(f"本地视频加载成功: {os.path.basename(video_path)}")
    elif use_local and local_path.strip():
        st.warning(f"文件不存在: {local_path.strip()}")

    if video_path:
        # 显示视频预览
        st.markdown("### 🎬 视频预览")
        st.video(video_path)
        input_orientation = detect_orientation(video_path)
        landscape_resolution_choice = get_landscape_resolution_choice()
        if input_orientation == "portrait":
            st.info("输出分辨率：竖屏视频按项目原规范输出 1080x1920")
        else:
            landscape_resolution_choice = st.selectbox(
                "输出分辨率",
                list(LANDSCAPE_RESOLUTION_CHOICES),
                index=list(LANDSCAPE_RESOLUTION_CHOICES).index(landscape_resolution_choice),
                format_func=lambda x: RESOLUTION_LABELS.get(x, x),
                key="subtitle_landscape_resolution_choice",
            )
            shared_config = load_slice_config()
            shared_config["landscape_resolution_choice"] = landscape_resolution_choice
            st.session_state.slice_config = shared_config
            save_slice_config(shared_config)
        
        # 开始处理按钮
        if st.button("🚀 开始处理", use_container_width=True):
            st.session_state.processing = True
            st.session_state.process_logs = []
            
            # 显示处理状态
            status_placeholder = st.empty()
            logs_placeholder = st.empty()
            
            def add_log(msg):
                st.session_state.process_logs.append(msg)
                logs_placeholder.text_area("处理日志", value="\n".join(st.session_state.process_logs), height=200)
            
            try:
                add_log("正在提取音频...")
                audio, sr = extract_audio_from_video(video_path)
                add_log(f"音频提取完成，时长: {len(audio)/sr:.1f}秒")
                
                add_log("正在使用豆包API识别文字...")
                asr_result = run_asr(audio, sr, do_preprocess=st.session_state.preprocessing)
                
                if asr_result.get("error"):
                    raise Exception(asr_result["error"])
                
                add_log(f"识别完成，总时间: {asr_result['total_time']:.2f}秒")
                add_log(f"识别文本: {asr_result['text'][:100]}...")
                
                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_name = f"{base_name}_subtitled.mp4"
                output_path = os.path.join(temp_dir, output_name)
                
                add_log("正在检测视频方向...")
                orientation = detect_orientation(video_path)
                add_log(f"视频方向: {orientation}")
                landscape_resolution_choice = get_landscape_resolution_choice()
                output_spec = resolve_output_resolution_spec(orientation, landscape_resolution_choice)
                add_log(f"输出分辨率: {output_spec.label}")
                
                add_log("正在生成字幕并烧录到视频...")
                success, result = generate_video_with_subtitles(
                    video_path,
                    asr_result,
                    output_path,
                    output_spec=output_spec,
                    orientation=orientation,
                    subtitle_mode=st.session_state.subtitle_mode
                )
                
                if success:
                    add_log(f"处理完成! 输出文件: {output_name}")
                    st.session_state.last_result = result
                    
                    # 显示处理结果
                    st.markdown("### 🎉 处理完成")
                    st.success(f"字幕生成成功！")
                    
                    # 显示输出视频
                    st.markdown("### 📺 输出视频")
                    st.video(output_path)
                    
                    # 提供下载按钮
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="💾 下载视频",
                            data=f,
                            file_name=output_name,
                            mime="video/mp4",
                            use_container_width=True
                        )
                else:
                    raise Exception(result)
                    
            except Exception as e:
                add_log(f"错误: {str(e)}")
                st.error(f"处理失败: {str(e)}")
            finally:
                st.session_state.processing = False


def main():
    """主函数：双 Tab（切片 + 字幕）。"""
    render_header()

    tab1, tab2 = st.tabs(["🎬 视频切片", "🎤 字幕生成"])

    with tab1:
        render_slicing_mode()

    with tab2:
        render_sidebar()
        render_subtitle_mode()

    # 底部信息
    st.markdown("""
    <hr>
    <div style="text-align: center; opacity: 0.7;">
        <p>演唱会工具台 | 视频切片 + 豆包字幕生成</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
