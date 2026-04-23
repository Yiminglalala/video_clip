"""
FunASR + LRC 歌词匹配流水线 v4（最终版）
- 正确LRC解析（多时间戳行）
- 字幕重叠（持续到下一行）
- FFmpeg: 复制ASS到视频目录避免路径问题
"""
import os, sys, json, re, shutil, subprocess
os.environ["HF_HOME"] = r"D:\video_clip\.cache\huggingface"

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from difflib import SequenceMatcher
from funasr import AutoModel

def fmt_t(s):
    h=int(s//3600); m=int((s%3600)//60); sec=int(s%60); cs=int((s%1)*100)
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

def parse_lrc(raw_text):
    """正确处理多时间戳LRC行，如 [03:44.96][00:16.98]你哭了吗"""
    entries = []
    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('[00:00.00]'):
            continue
        
        # 提取所有时间戳 [mm:ss.xx] 或 [ss.xx]
        times = []
        text_part = line
        while True:
            m = re.match(r'\[(\d{1,2}):(\d{2})\.(\d{2,3})\](.*)', text_part)
            if not m:
                break
            mm, ss, ms = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if len(m.group(3)) <= 2: ms *= 10
            t = mm*60 + ss + ms/1000.0
            times.append(t)
            text_part = m.group(4)  # 剩余部分继续匹配
        
        text = text_part.strip()
        if not text:
            continue
        # 跳过元数据行
        low = text.lower()
        if any(k in low for k in ['作词','作曲','编曲','arranged']):
            continue
        # 用最后一个时间戳（通常是最精确的歌词时间）
        if times:
            # 如果有多个时间戳，取最小的那个（通常是当前行的时间）
            best_time = min(times)
            entries.append((round(best_time, 2), text))
    
    # 按时间排序+去重
    entries.sort(key=lambda x: x[0])
    seen = set()
    unique = []
    for t, l in entries:
        if l not in seen:
            seen.add(l)
            unique.append((t, l))
    return unique

# ========== 配置 ==========
VIDEO = r"D:\video_clip\output\A-Lin_声音梦境线上音乐会_20260412_1157\Song_01\Song_01_主歌_00m15s-00m59s.mp4"
OUT_DIR = r"D:\video_clip\output\lyric_aligned"
AUDIO_PATH = os.path.join(OUT_DIR, "audio.wav")
OUTPUT_VIDEO = os.path.join(OUT_DIR, "huangtang_funasr_v4.mp4")

LRC_RAW = """[03:44.96][00:16.98]你哭了吗 我听不到你说话
[03:52.19][00:24.07]转身走吧 没有必要再勉强
[01:56.03][00:29.97]只是输给了一个诚实的谎话
[00:37.15]我们怎么会经不起
[00:40.79]背叛的冲刷
[00:45.52]你失望吗 我并不是你想象
[00:52.61]剩下什么 可以用爱伪装
[00:58.60]原谅不是唯一结束问题的回答
[01:05.81]我真的开始怀疑 爱情的重量
[03:08.02][02:10.73][01:13.10]终于 让我看穿了爱情
[03:12.93][02:15.67][01:18.28]我明白这场游戏 输的五体投地
[03:18.55][02:21.01][01:23.67]关于你布下的局
[03:22.34][02:24.55][01:27.23]终于 我承认了我伤心
[03:27.18][02:29.89][01:32.65]我确定把这回忆 抹的干干净净
[03:33.01][01:38.51]收拾你的荒唐 然后离去
[01:45.32]可不可以让自己逃离
[01:52.45]用最后的力气
[02:03.09]我们怎么会爱上
[02:07.12]彼此的荒唐
[02:35.76]收拾你的荒唐 oh~
[02:40.09]一幕幕 我闭不上眼睛
[02:42.95]残忍的甜蜜
[02:47.04]一封封 删不去的简讯
[02:50.08]烙在心里 藏在心里
[02:54.15]我们爱过的遐想 无法释放
[03:00.79]我害怕我不忍心再说一句 我恨你~~"""

LRC_LINES = parse_lrc(LRC_RAW)
print(f"Parsed {len(LRC_LINES)} LRC lines:")
for t, l in LRC_LINES:
    print(f"  [{t:.2f}s] {l}")

# ===== Step 1: 音频 =====
print("\n" + "=" * 60)
assert os.path.exists(VIDEO), f"Video missing: {VIDEO}"
probe = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
    "-of","default=noprint_wrappers=1:nokey=1", VIDEO], capture_output=True, text=True)
vid_dur = float(probe.stdout.strip())
print(f"Video: {os.path.basename(VIDEO)}, {vid_dur:.1f}s")

print("Step 1: Audio extract")
subprocess.run(["ffmpeg","-y","-i",VIDEO,"-vn","-acodec","pcm_s16le",
    "-ar","16000","-ac","1", AUDIO_PATH], check=True, capture_output=True)

y, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
sos = butter(4, 100, btype='high', fs=sr, output='sos')
y = sosfilt(sos, y)
peak = np.max(np.abs(y))
if peak > 0: y = y * 0.9 / peak
audio_np = y.astype(np.float32)
dur = len(audio_np)/sr
print(f"Audio: {dur:.1f}s @ {sr}Hz")

# ===== Step 2: FunASR =====
print("\nStep 2: FunASR...")
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc-c", device="cuda:0")
res = model.generate(input=audio_np, cache={}, language="zh", use_itn=True, batch_size_s=300)
asr_text = res[0].get("text", "")
ts_list = res[0].get("timestamp", [])
print(f"ASR: {asr_text}")
print(f"Tokens: {len(ts_list)}")

# ===== Step 3: 词级信息 =====
print("\nStep 3: Word-level")
wi = []
chars = list(asr_text)
if len(ts_list) >= len(chars):
    for i,c in enumerate(chars):
        wi.append({"c":c,"s":ts_list[i][0]/1000,"e":ts_list[i][1]/1000})
else:
    r = len(chars)/max(len(ts_list),1)
    for i,c in enumerate(chars):
        ti=int(i/r)
        wi.append({"c":c,"s":ts_list[ti][0]/1000,"e":ts_list[ti][1]/1000} if ti<len(ts_list) else {"c":c,"s":wi[-1]["e"],"e":wi[-1]["e"]+.3})
print(f"Words: {len(wi)}")

# ===== Step 4: LRC 匹配 =====
print("\nStep 4: LRC match")
def cln(t): return t.replace(" ","").replace("，",",").replace("。",".").replace("？","?").replace("、","").replace("~","")

def match_one(asr_clean, lrc_line):
    lc = cln(lrc_line); bp=-1; br=0; be=0; ml=min(max(2,len(lc)//2),6)
    for s in range(len(asr_clean)):
        for e in range(s+ml, min(len(asr_clean),s+len(lc)+5)):
            r=SequenceMatcher(None,lc,asr_clean[s:e]).ratio()
            if r>br: bp=s;br=r;be=e
    return {"line":lrc_line,"pos":bp,"ratio":br,"match":asr_clean[bp:be] if bp>=0 else ""}

ac = cln(asr_text)
matches = []
for lt, ll in LRC_LINES:
    m = match_one(ac, ll)
    matches.append(m)
    st = "OK" if m["ratio"]>0.35 else "--"
    print(f"  [{st}] {m['ratio']:.2f} | {ll[:30]}")
    if m["ratio"] > 0.25:
        print(f"       -> '{m['match']}'")

# ===== Step 5: 字幕生成（重叠 + 去重 + 高阈值）=====
print("\nStep 5: Subtitles (overlap + dedup)")
CONFIDENCE_MIN = 0.50  # 提高最低置信度阈值（之前0.30太低）
OVERLAP_TOLERANCE = 1.0  # 时间重叠超过1秒则视为重复

subs_raw = []
for mi,m in enumerate(matches):
    if m["ratio"] < CONFIDENCE_MIN or m["pos"] < 0:
        continue
    pos = m["pos"]
    st = wi[pos]["s"]
    
    # 找下一行的开始时间
    nxt = None
    for mj in range(mi+1, len(matches)):
        n = matches[mj]
        if n["ratio"] >= CONFIDENCE_MIN and 0 <= n["pos"] < len(wi):
            nxt = wi[n["pos"]]["s"]
            break
    
    et = (dur + 0.5) if nxt is None else nxt
    if st >= et: et = st + max(dur / 10, 2.0)
    
    subs_raw.append({
        "start": round(st, 2),
        "end": round(et, 2),
        "text": m["line"],
        "conf": round(m["ratio"], 2),
        "raw_idx": mi,
    })

# === 去重叠：按时间排序，移除与高置信度行重叠的低置信度行 ===
subs_raw.sort(key=lambda x: x["start"])
subs = []
for candidate in subs_raw:
    overlaps_existing = False
    for existing in subs:
        # 检查时间重叠
        start_overlap = max(candidate["start"], existing["start"])
        end_overlap = min(candidate["end"], existing["end"])
        overlap_sec = max(0, end_overlap - start_overlap)
        
        if overlap_sec > OVERLAP_TOLERANCE:
            # 有重叠 → 只保留置信度更高的那个
            if candidate["conf"] > existing["conf"]:
                # 新的更好，替换旧的
                subs.remove(existing)
                subs.append(candidate)
            # 否则丢弃候选行
            overlaps_existing = True
            break
    
    if not overlaps_existing:
        subs.append(candidate)

# 最后按时间再排一次序
subs.sort(key=lambda x: x["start"])

if len(subs) < 2:
    print("[Fallback]")
    subs=[]; ci=0
    for p in re.split(r'[，。！？、；]',asr_text):
        p=p.strip(); pc=len(p)
        if not pc: continue
        if ci<len(wi):
            s=wi[ci]["s"]; ei=min(ci+pc+1,len(wi)); e=wi[ei-1]["e"]
            nci=ci+pc+1
            if nci<len(wi): e=wi[nci]["s"]
            subs.append({"start":round(s,2),"end":round(e,2),"text":p,"conf":0.5})
        ci+=pc+1

print(f"\nMatched {len(subs)} lines:")
for sl in subs:
    d=sl["end"]-sl["start"]
    print(f"  [{sl['start']:.2f}-{sl['end']:.2f}] ({d:.1f}s) {sl['text']}")

# ===== Step 6: ASS 文件 =====
print("\nStep 6: ASS")
ass_path = os.path.join(OUT_DIR, "subtitle_v4.ass")
ass = f"""[Script Info]
Title: A-Lin 荒唐 v4
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,3,2,40,40,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
for sl in subs:
    ass += f"Dialogue: 0,{fmt_t(sl['start'])},{fmt_t(sl['end'])},Default,,0,0,0,,{sl['text']}\n"
with open(ass_path,"w",encoding="utf-8-sig") as f: f.write(ass)
print(f"Saved: {ass_path}")

# ===== Step 7: FFmpeg（关键修复：复制ASS到视频目录）=====
print("\nStep 7: FFmpeg burn")
vid_dir = os.path.dirname(os.path.abspath(VIDEO))
vname = os.path.basename(VIDEO)
aname_local = "subtitle_v4.ass"
local_ass_path = os.path.join(vid_dir, aname_local)

# 复制ASS到视频同目录（解决Windows路径中冒号被解析为尺寸的问题）
shutil.copy2(ass_path, local_ass_path)
print(f"Copied ASS to video dir: {local_ass_path}")

cmd = f'cd "{vid_dir}" && ffmpeg -y -i "{vname}" -vf "ass={aname_local}" -c:v libx264 -preset medium -crf 18 -c:a copy "{os.path.abspath(OUTPUT_VIDEO)}"'
result = subprocess.run(cmd, shell=True, capture_output=True, encoding='utf-8', errors='replace')

# 清理临时ASS
try: os.remove(local_ass_path)
except: pass

if result.returncode != 0:
    err = result.stderr[-600:] if result.stderr else "(empty)"
    print(f"FFmpeg error {result.returncode}:\n{err}")
else:
    sz = os.path.getsize(OUTPUT_VIDEO)/(1024*1024)
    print(f"\n{'='*60}")
    print(f"[OK] {OUTPUT_VIDEO} ({sz:.1f}MB)")
    print(f"{'='*60}")
    print(f"\nSubtitles ({len(subs)} lines):")
    for sl in subs:
        d=sl["end"]-sl["start"]
        print(f"  [{sl['start']:.2f}-{sl['end']:.2f}] ({d:.1f}s displayed) {sl['text']}")

report={"method":"FunASR+LRC v4","video":VIDEO,"dur_s":round(vid_dur,1),
    "asr":asr_text,"tokens":len(ts_list),"matched":len(subs),"subs":subs}
with open(os.path.join(OUT_DIR,"funasr_report_v4.json"),"w",encoding="utf-8") as f:
    json.dump(report,f,ensure_ascii=False,indent=2)
