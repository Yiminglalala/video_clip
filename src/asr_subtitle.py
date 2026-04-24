# -*- coding: utf-8 -*-
"""
ASR 字幕模块 - 与 app.py 保持完全一致
"""

import os
import tempfile
import uuid
from typing import List, Dict, Optional, Tuple

from src.output_spec import (
    DEFAULT_LANDSCAPE_RESOLUTION,
    OutputResolutionSpec,
    SUBTITLE_FONT_FAMILY,
    build_cover_crop_filter,
    build_ass_filter_value,
    resolve_output_resolution_spec,
    STANDARD_VIDEO_OUTPUT_ARGS,
)


# ============================================================================
# Whisper 模型管理
# ============================================================================

_whisper_model = None
_whisper_model_name = None


def get_whisper_model(model_name: str = "small"):
    global _whisper_model, _whisper_model_name
    if _whisper_model is None or _whisper_model_name != model_name:
        import whisper
        print(f"[Whisper] 加载模型: {model_name} ...")
        _whisper_model = whisper.load_model(model_name)
        _whisper_model_name = model_name
        print(f"[Whisper] 模型加载完成")
    return _whisper_model


def transcribe_with_timestamps(
    audio, sr, model_name: str = "small", language: str = "zh"
):
    try:
        import whisper
        model = get_whisper_model(model_name)
        if sr != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
        result = model.transcribe(audio_16k, language=language, word_timestamps=True, verbose=False)
        words = []
        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                words.append({"word": word["word"].strip(), "start": float(word["start"]), "end": float(word["end"])})
        sentences = []
        for seg in result.get("segments", []):
            sentences.append({"text": seg["text"].strip(), "start": float(seg["start"]), "end": float(seg["end"])})
        return {"words": words, "sentences": sentences, "full_text": result.get("text", "")}
    except Exception:
        from src.lyric_subtitle import transcribe_clip_with_whisperx
        rough = transcribe_clip_with_whisperx(audio, sr, model_name=model_name)
        segments = rough.get("segments", [])
        return {"words": [], "sentences": [{"text": seg.get("text", "").strip(), "start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0))} for seg in segments if seg.get("text")], "full_text": rough.get("text", "")}


# ============================================================================
# ASS 字幕生成（与 app.py 完全一致）
# ============================================================================

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
    events = []
    expanded_sentences = []
    for sent in sentences:
        expanded_sentences.extend(_split_long_sentence_entry(sent, max_chars=14))
    for i, sent in enumerate(expanded_sentences):
        start = sent.get("start", 0)
        end = sent.get("end", start + 3)
        text = sent.get("text", "")
        
        def format_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h}:{m:02d}:{s:05.2f}"
        
        start_str = format_time(start)
        end_str = format_time(end)
        text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N").replace("\r", "")
        # 自动换行处理
        events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0000,0000,0000,,{text}")
    
    if output_spec is None:
        output_spec = resolve_output_resolution_spec(orientation, DEFAULT_LANDSCAPE_RESOLUTION)
    play_res_x = output_spec.width
    play_res_y = output_spec.height
    font_size = 90
    
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
    events = []
    for i, word in enumerate(words):
        start = word.get("start", 0)
        end = word.get("end", start + 1)
        text = word.get("text", word.get("word", ""))
        
        def format_time(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h}:{m:02d}:{s:05.2f}"
        
        start_str = format_time(start)
        end_str = format_time(end)
        text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N").replace("\r", "")
        events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0000,0000,0000,,{text}")
    
    if output_spec is None:
        output_spec = resolve_output_resolution_spec(orientation, DEFAULT_LANDSCAPE_RESOLUTION)
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
""" + events_text
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ass_content)


def words_to_sentence_chunks(words, max_chars=24, max_gap=0.6):
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
        need_break = (gap > max_gap or len(current_text) + len(text) > max_chars or
                     (current_text and current_text[-1] in "。！？!?,，、；;：:"))
        if need_break:
            flush_chunk()
            current_start = start
            current_end = end
            current_text = text
        else:
            if current_text and current_text[-1].isascii() and current_text[-1].isalnum() and text[0].isascii() and text[0].isalnum():
                current_text += " "
            current_text += text
            current_end = end
        last_end = end
    flush_chunk()
    return chunks


def _format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text):
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _escape_srt_text(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt_from_words(words, output_path):
    sentences = words_to_sentence_chunks(words)
    lines = []
    for i, sent in enumerate(sentences, 1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(sent['start'])} -> {_format_srt_time(sent['end'])}")
        lines.append(sent["text"])
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    print(f"[ASR] SRT 字幕已生成: {output_path}")
    return output_path


def _merge_duplicate_words(lines, style_name):
    result = []
    for line in lines:
        if not line.startswith("Dialogue:"):
            result.append(line)
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            result.append(line)
            continue
        start, end = parts[1], parts[2]
        text = parts[9].strip()
        if (result and result[-1].startswith("Dialogue:") and result[-1].endswith(text)):
            last_parts = result[-1].split(",", 9)
            if len(last_parts) >= 10:
                last_parts[2] = end
                result[-1] = ",".join(last_parts)
                continue
        result.append(line)
    return result


# ============================================================================
# 完整流水线（与 app.py 保持一致）
# ============================================================================

def process_subtitles(
    video_path,
    output_path,
    audio_segment=None,
    sr=24000,
    orientation="auto",
    output_spec: OutputResolutionSpec | None = None,
    model_name="small",
    subtitle_mode="word",
    use_nvenc=True,
    prefer_lyrics=True,
    song_hint=None,
):
    import subprocess
    print(f"[ASR Subtitle] 开始处理: {video_path}")
    
    if audio_segment is None:
        print("[ASR Subtitle] 提取音频...")
        import librosa
        audio_segment, _ = librosa.load(video_path, sr=sr, mono=True)
        print(f"[ASR Subtitle] 音频长度: {len(audio_segment)/sr:.1f}s")
    
    result = None
    if isinstance(song_hint, dict):
        lyrics_requested = bool(prefer_lyrics and (song_hint.get("title") or song_hint.get("lyrics_text")))
    else:
        lyrics_requested = bool(prefer_lyrics and song_hint)
    if prefer_lyrics:
        try:
            from src.lyric_subtitle import build_lyric_subtitle_result
            print("[ASR Subtitle] 尝试标准歌词字幕...")
            result = build_lyric_subtitle_result(
                audio_segment=audio_segment, sr=sr, model_name="tiny", song_hint=song_hint,
                auto_locate=bool(song_hint.get("auto_locate", True)) if isinstance(song_hint, dict) else True,
                allow_song_identification=False,
            )
            if result:
                subtitle_mode = "sentence"
                print("[ASR Subtitle] 歌词链路命中")
        except Exception as e:
            print(f"[ASR Subtitle] 标准歌词链路失败: {e}")
    
    if result is None and lyrics_requested:
        return False, "标准歌词未找到或未确认，请在 lyrics_hints.json 中提供 lyrics_text/start_line，或关闭标准歌词模式。"
    
    if result is None:
        print(f"[ASR Subtitle] Whisper {model_name} 推理中...")
        result = transcribe_with_timestamps(audio_segment, sr, model_name=model_name, language="auto")
    
    words = result["words"]
    sentences = result["sentences"]
    
    if not words and not sentences:
        print("[ASR Subtitle] 未检测到语音，跳过字幕")
        import shutil
        shutil.copy2(video_path, output_path)
        return True, output_path
    
    print(f"[ASR Subtitle] 识别词数: {len(words)}, 句子数: {len(sentences)}")
    
    temp_dir = tempfile.gettempdir()
    ass_path = os.path.join(temp_dir, f"temp_subtitle_{uuid.uuid4().hex}.ass")
    if output_spec is None:
        output_spec = resolve_output_resolution_spec(orientation, DEFAULT_LANDSCAPE_RESOLUTION)
    if subtitle_mode == "word":
        generate_ass_from_words(words, ass_path, output_spec=output_spec, orientation=orientation)
    else:
        generate_ass_from_sentences(sentences, ass_path, output_spec=output_spec, orientation=orientation)
    
    print(f"[ASR Subtitle] 烧录到视频...")
    target_w = output_spec.width
    target_h = output_spec.height
    video_filter = build_cover_crop_filter(target_w, target_h, extra_filter=build_ass_filter_value(ass_path))
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    same_output_path = os.path.abspath(video_path) == os.path.abspath(output_path)
    working_output_path = os.path.join(temp_dir, f"temp_subtitled_{uuid.uuid4().hex}.mp4") if same_output_path else output_path
    
    def _build_cmd(video_codec_args, out_path):
        return [
            "ffmpeg", "-y", "-i", video_path,
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", video_filter,
            *video_codec_args,
            "-c:a", "aac", "-b:a", "160k", "-ac", "2",
            "-r", "60",
            *STANDARD_VIDEO_OUTPUT_ARGS,
            "-movflags", "+faststart",
            out_path,
        ]
    
    def _run_cmd(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=300)
    
    nvenc_args = ["-c:v", "h264_nvenc", "-rc", "cbr", "-b:v", "5M", "-maxrate", "5M", "-bufsize", "10M", "-preset", "p4"]
    cpu_args = ["-c:v", "libx264", "-crf", "23", "-preset", "medium"]
    
    try:
        run_result = None
        encoder_name = "libx264"
        if use_nvenc:
            cmd = _build_cmd(nvenc_args, working_output_path)
            run_result = _run_cmd(cmd)
            encoder_name = "h264_nvenc"
            if run_result.returncode != 0:
                print(f"[ASR Subtitle] NVENC 失败，回退 CPU:\n{run_result.stderr[-500:]}")
        
        if run_result is None or run_result.returncode != 0:
            cmd = _build_cmd(cpu_args, working_output_path)
            run_result = _run_cmd(cmd)
            encoder_name = "libx264"
        
        if run_result.returncode != 0:
            print(f"[ASR Subtitle] FFmpeg 错误:\n{run_result.stderr[-500:]}")
            print("[ASR Subtitle] 尝试软字幕模式...")
            ok, msg = _burn_subtitle_soft(video_path, ass_path, working_output_path)
            if not ok:
                return False, msg
        
        if same_output_path:
            os.replace(working_output_path, output_path)
        
        print(f"[ASR Subtitle] 完成: {output_path} (encoder={encoder_name})")
        return True, output_path
    except subprocess.TimeoutExpired:
        return False, "FFmpeg 超时（>5分钟）"
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)
        if same_output_path and os.path.exists(working_output_path):
            try:
                os.remove(working_output_path)
            except OSError:
                pass


def _burn_subtitle_soft(video_path, ass_path, output_path):
    import subprocess
    temp_mp4 = os.path.join(tempfile.gettempdir(), f"temp_video_only_{uuid.uuid4().hex}.mp4")
    cmd_extract = ["ffmpeg", "-y", "-i", video_path, "-c:v", "copy", "-an", temp_mp4]
    r = subprocess.run(cmd_extract, capture_output=True)
    if r.returncode != 0:
        return False, f"提取视频流失败: {r.stderr[-200:]}"
    cmd_merge = ["ffmpeg", "-y", "-i", temp_mp4, "-i", ass_path, "-c:v", "copy", "-c:s", "mov_text", "-c:a", "copy", "-metadata:s:s:0", "language=chi", output_path]
    r = subprocess.run(cmd_merge, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if os.path.exists(temp_mp4):
        os.remove(temp_mp4)
    if r.returncode != 0:
        return False, f"合并字幕失败: {r.stderr[-200:]}"
    print(f"[ASR Subtitle] 软字幕模式完成: {output_path}")
    return True, output_path


# ============================================================================
# 调试工具
# ============================================================================

def diagnose_audio(audio, sr):
    import librosa
    import numpy as np
    duration = len(audio) / sr
    rms = librosa.feature.rms(y=audio, hop_length=1024)[0]
    peak_db = float(20 * np.log10(np.max(np.abs(audio)) + 1e-8))
    mean_db = float(20 * np.log10(np.mean(np.abs(audio)) + 1e-8))
    silence_threshold = np.percentile(rms, 20)
    silence_ratio = float(np.mean(rms < silence_threshold))
    sr_valid = sr >= 16000
    return {
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "peak_db": round(peak_db, 1),
        "mean_db": round(mean_db, 1),
        "silence_ratio": round(silence_ratio * 100, 1),
        "sr_valid_for_whisper": sr_valid,
        "has_audio": mean_db > -50,
    }
