"""
LRC歌词强制对齐流水线 v3
核心思路:
  1. LRC = 正确的歌词文本 + 原始时间轴(近似)
  2. Whisper word-level = 音频中每个词的精确时间戳
  3. 用DTW将LRC行匹配到Whisper词序列 → 得到每行歌词的精确起止时间
  4. 输出ASS字幕 + 烧录视频
"""

import os, sys, re, json, subprocess, warnings
import numpy as np
from difflib import SequenceMatcher
warnings.filterwarnings("ignore")

# === 配置 ===
VIDEO_PATH = r"D:\video_clip\output\A-Lin_声音梦境线上音乐会_20260412_1157\Song_01\Song_01_副歌_01m12s-01m41s.mp4"
LRC_PATH   = r"D:\video_clip\output\lyric_aligned\huangtang.lrc"
OUT_DIR    = r"D:\video_clip\output\lyric_aligned"
OUT_VIDEO  = os.path.join(OUT_DIR, "huangtang_v3_subtitled.mp4")
ASS_PATH   = os.path.join(OUT_DIR, "subtitle_v3.ass")
REPORT     = os.path.join(OUT_DIR, "report_v3.json")
LOG_PATH   = os.path.join(OUT_DIR, "v3_pipeline.log")

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    """写日志到文件，同时stdout"""
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

# 清空旧日志
if os.path.exists(LOG_PATH): os.remove(LOG_PATH)
log("="*60)
log("=== LRC Forced Alignment Pipeline v3 ===")
log(f"Video: {VIDEO_PATH}")
log(f"LRC:   {LRC_PATH}")
log("="*60)


# ============================================================
# Step 1: 提取音频
# ============================================================
AUDIO_PATH = os.path.join(OUT_DIR, "audio.wav")
log("\n[Step1] Extracting audio...")
r = subprocess.run([
    "ffmpeg", "-y", "-i", VIDEO_PATH,
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    AUDIO_PATH
], capture_output=True)
assert r.returncode == 0 and os.path.exists(AUDIO_PATH), f"FFmpeg failed: {r.stderr.decode(errors='replace')[:200]}"
duration_s = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",AUDIO_PATH], 
                                   capture_output=True, text=True).stdout.strip())
log(f"  Audio: {duration_s:.1f}s @ 16kHz mono")


# ============================================================
# Step 2: 解析LRC歌词
# ============================================================
log("\n[Step2] Parsing LRC...")

def parse_lrc(path):
    """解析LRC文件，返回 [(time_seconds, text), ...]"""
    entries = []
    ts_re = re.compile(r'\[(\d{2}):(\d{2})\.(\d{2,3})\](.+)')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = ts_re.match(line.strip())
            if m and not any(line.strip().startswith(x) for x in ['[ti:', '[ar:', '[al:', '[by:', '[offset:', '[00:00]']):
                mm, ss, ms = float(m.group(1)), float(m.group(2)), float(m.group(3))
                # 处理毫秒位数 (可能是2位或3位)
                if ms >= 100:
                    ts = mm * 60 + ss + ms / 1000.0
                else:
                    ts = mm * 60 + ss + ms / 100.0
                text = m.group(4).strip()
                # 过滤掉非歌词行
                if text and not text.startswith('作曲') and not text.startswith('作词') and text != '纯音乐，请欣赏':
                    entries.append((ts, text))
    return entries

lrc_lines = parse_lrc(LRC_PATH)
log(f"  Parsed {len(lrc_lines)} LRC lines:")
for i, (ts, txt) in enumerate(lrc_lines):
    log(f"    [{i}] [{ts:.2f}s] {txt}")


# ============================================================
# Step 3: Whisper 词级ASR (word-level timestamps)
# ============================================================
log("\n[Step3] Running Whisper word-level ASR...")
import whisper
model = whisper.load_model("small")

result = model.transcribe(
    AUDIO_PATH,
    language="Chinese",
    verbose=False,
    word_timestamps=True,       # 关键！获取每个词的时间戳
)

# 收集所有词及其时间戳
whisper_words = []  # [{"word": str, "start": float, "end": float}, ...]
for seg in result["segments"]:
    if "words" in seg:
        for w in seg["words"]:
            whisper_words.append({
                "text": w.get("word", "").strip(),
                "start": w["start"],
                "end": w["end"],
            })
    elif "text" in seg:
        # fallback: 没有词级信息时用segment级
        whisper_words.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
        })

log(f"  Whisper found {len(whisper_words)} word-level tokens:")
for w in whisper_words[:15]:
    log(f"    [{w['start']:.2f}-{w['end']:.2f}s] '{w['text']}'")
if len(whisper_words) > 15:
    log(f"    ... +{len(whisper_words)-15} more")


# ============================================================
# Step 4: LRC → Whisper 时间轴对齐 (核心算法!)
# ============================================================
log("\n[Step4] Aligning LRC to Whisper timeline...")

def chinese_char_similarity(c1, c2):
    """中文单字相似度：相同=1, 不同=0"""
    return 1.0 if c1 == c2 else 0.0

def align_line_to_words(lrc_text, all_words, start_search_from=0):
    """
    将一行LRC歌词对齐到Whisper词序列上。
    
    方法: 在所有词的拼接文本中搜索最相似的子序列，
    返回匹配到的词索引范围和实际时间范围。
    """
    lrc_chars = re.sub(r'[\s，。、？！…—·\-\(\)\[\]]', '', lrc_text)
    if len(lrc_chars) < 2:
        return None
    
    # 构建所有Whisper词的字符序列（带索引映射）
    char_list = []
    word_idx_for_char = []
    for wi, w in enumerate(all_words[start_search_from:], start=start_search_from):
        wc = re.sub(r'[\s，。、？！…—·\-\(\)\[\]]', '', w['text'])
        for ch in wc:
            char_list.append(ch)
            word_idx_for_char.append(wi)
    
    if not char_list:
        return None
    
    # 用滑动窗口+编辑距离找最佳匹配
    best_score = -1
    best_start = 0
    best_end = min(len(char_list), len(lrc_chars) + 5)
    window_size = max(len(lrc_chars) - 2, min(len(lrc_chars) + 5, 20))  # 允许一定误差
    
    search_end = len(char_list)
    for start in range(min(search_end, len(char_list))):
        end = min(start + window_size, search_end)
        subseq = ''.join(char_list[start:end])
        
        score = SequenceMatcher(None, lrc_chars, subseq).ratio()
        if score > best_score:
            best_score = score
            best_start = start
            best_end = end
        
        # 不需要搜太远（超过3倍长度就停）
        if start > 0 and (start - start_search_from) > len(lrc_chars) * 3:
            break
    
    if best_score < 0.25:  # 相似度太低
        return None
    
    # 获取匹配到的时间范围
    first_wi = word_idx_for_char[best_start] if best_start < len(word_idx_for_char) else -1
    last_wi = word_idx_for_char[min(best_end, len(word_idx_for_char)-1)] if best_end > 0 else -1
    
    if first_wi == -1 or last_wi == -1:
        return None
    
    t_start = all_words[first_wi]["start"]
    t_end = all_words[last_wi]["end"]
    
    return {
        "lrc_text": lrc_text,
        "matched_whisper_text": ''.join(char_list[best_start:best_end]),
        "score": round(best_score, 3),
        "time_start": round(t_start, 2),
        "time_end": round(t_end, 2),
        "first_word_idx": first_wi,
        "last_word_idx": last_wi,
    }

# 对每一行LRC进行对齐
aligned_lines = []
search_from = 0
for i, (lrc_time, lrc_text) in enumerate(lrc_lines):
    result = align_line_to_words(lrc_text, whisper_words, start_search_from=search_from)
    if result:
        aligned_lines.append(result)
        search_from = result["last_word_idx"] + 1  # 下一次从这行之后开始搜
        status = f"[{result['time_start']:.2f}-{result['time_end']:.2f}s] score={result['score']}"
        log(f"  [OK] [{i}] {status} | LRC:'{lrc_text}' → match:'{result['matched_whisper_text']}'")
    else:
        log(f"  [??] [{i}] FAILED to match: '{lrc_text}'")
        # fallback: 用LRC原始时间
        if i + 1 < len(lrc_lines):
            next_t = lrc_lines[i+1][0]
        else:
            next_t = lrc_time + 5
        aligned_lines.append({
            "lrc_text": lrc_text,
            "matched_whisper_text": "",
            "score": 0,
            "time_start": round(lrc_time, 2),
            "time_end": round(next_t, 2),
            "first_word_idx": -1,
            "last_word_idx": -1,
        })


# ============================================================
# Step 5: 生成ASS字幕
# ============================================================
log("\n[Step5] Generating ASS subtitle...")

def make_ass(aligned_lines, video_duration, font_size=36):
    width, height = 1920, 1080
    ass = [
        "[Script Info]",
        "Title: Lyric Subtitle",
        "ScriptType: 4.00",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H66000000,-1,0,0,0,100,100,0,0,1,2,2,2,30,30,50,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    
    for al in aligned_lines:
        s = format_time_ass(al["time_start"])
        e = format_time_ass(al["time_end"])
        text = al["lrc_text"]
        ass.append(f"Dialogue: 0,{s},{e},Default,,0,0,0,,{text}")
    
    return "\n".join(ass)

def format_time_ass(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"

# 从视频获取真实时长
vid_dur_str = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","format=duration",
                              "-of","csv=p=0",VIDEO_PATH], capture_output=True, text=True).stdout.strip()
video_dur = float(vid_dur_str) if vid_dur_str else duration_s

ass_content = make_ass(aligned_lines, video_dur)
with open(ASS_PATH, 'w', encoding='utf-8-sig') as f:
    f.write(ass_content)
log(f"  ASS saved: {ASS_PATH} ({len(aligned_lines)} lines)")
for al in aligned_lines:
    log(f"    [{al['time_start']:.2f}-{al['time_end']:.2f}s] {al['lrc_text']} (match={al['score']})")


# ============================================================
# Step 6: FFmpeg烧录字幕
# ============================================================
log("\n[Step6] Burning subtitles into video...")
r = subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-vf", f"subtitles={ASS_PATH.replace(os.sep, '/')}:force_style='Fontsize=36,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1'",
    "-c:v", "h264_nvenc" if os.name == 'nt' else "libx264",
    "-preset", "p4" if os.name == 'nt' else "medium",
    "-cq", "23",
    "-c:a", "copy",
    OUT_VIDEO
], capture_output=True, encoding='utf-8', errors='replace')

if r.returncode != 0:
    # fallback CPU encoder
    log("  NVENC failed, trying libx264...")
    r = subprocess.run([
        "ffmpeg", "-y",
        "-i", VIDEO_PATH,
        "-vf", f"subtitles={ASS_PATH.replace(os.sep, '/')}:force_style='Fontsize=36,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        OUT_VIDEO
    ], capture_output=True, encoding='utf-8', errors='replace')

if os.path.exists(OUT_VIDEO):
    size_mb = os.path.getsize(OUT_VIDEO) / (1024*1024)
    log(f"\n  OK! Output: {OUT_VIDEO} ({size_mb:.1f}MB)")
else:
    log(f"\n  FAIL! FFmpeg error: {r.stderr[-500:] if r.stderr else 'unknown'}")


# ============================================================
# 报告
# ============================================================
report = {
    "input_video": VIDEO_PATH,
    "lrc_file": LRC_PATH,
    "method": "Whisper-small word-level + LRC sliding-window alignment",
    "lrc_lines_total": len(lrc_lines),
    "aligned_lines": len(aligned_lines),
    "whisper_words": len(whisper_words),
    "lines": [{
        "lrc_text": al["lrc_text"],
        "time_start": al["time_start"],
        "time_end": al["time_end"],
        "match_score": al["score"],
        "whisper_match": al["matched_whisper_text"]
    } for al in aligned_lines],
    "output_video": OUT_VIDEO,
    "output_ass": ASS_PATH,
}
with open(REPORT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# 统计
matched = sum(1 for al in aligned_lines if al["score"] > 0.3)
log(f"\n{'='*60}")
log(f"DONE! {matched}/{len(aligned_lines)} lines well-matched (score>0.3)")
log(f"Report: {REPORT}")
log(f"{'='*60}")
