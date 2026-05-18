# -*- coding: utf-8 -*-
"""Step 2: 读取ASR结果 + LRC匹配 + 生成ASS + FFmpeg烧录"""
import os, sys, json, re, subprocess

VIDEO_PATH = r'D:\video_clip\output\A-Lin_声音梦境线上音乐会_20260412_1157\Song_01\Song_01_副歌_01m12s-01m41s.mp4'
LRC_PATH   = r'D:\video_clip\output\song_id_test\huangtang.lrc'
OUTPUT_DIR = r'D:\video_clip\output\lyric_aligned'
VIDEO_OFFSET = 72.0  # 视频从歌曲01:12开始

# 读取ASR结果
with open(os.path.join(OUTPUT_DIR, 'asr_results.json'), 'r', encoding='utf-8') as f:
    asr_results = json.load(f)
print(f"ASR segments: {len(asr_results)}")
for a in asr_results:
    print(f"  [{a['start']:.0f}-{a['end']:.0f}s] [{a['lang']}] \"{a['text'][:60]}\"")

# 读取视频时长
probe = subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_format', VIDEO_PATH],
                       capture_output=True)
duration = float(json.loads(probe.stdout)['format']['duration'])

# 解析LRC
lrc_lines = []
with open(LRC_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        times = re.findall(r'\[(\d+):(\d+\.\d+)\]', line)
        text = re.sub(r'\[\d+:\d+\.\d+\]', '', line).strip()
        text = re.sub(r'\s+', ' ', text)
        if text and times and not text.startswith(('作词','作曲','编曲')):
            for t in times:
                sec = int(t[0])*60 + float(t[1])
                lrc_lines.append({'time': sec, 'text': text})
lrc_lines.sort(key=lambda x: x['time'])

# 只保留视频范围内的LRC
lrc_in_video = [ll for ll in lrc_lines if 0 <= (ll['time'] - VIDEO_OFFSET) <= duration + 2]
print(f"\nLRC in video range: {len(lrc_in_video)}/{len(lrc_lines)}")
for ll in lrc_in_video:
    print(f"  [song {ll['time']:.0f}s -> video {ll['time']-VIDEO_OFFSET:.0f}s] {ll['text'][:50]}")

# Levenshtein匹配
def lev_ratio(a, b):
    if not a or not b: return 0.0
    a = re.sub(r'[^\u4e00-\u9fff\w]', '', a.lower())
    b = re.sub(r'[^\u4e00-\u9fff\w]', '', b.lower())
    if not a or not b: return 0.0
    m, n = len(a), len(b)
    if m > n: a, b = b, a; m, n = n, m
    dp = list(range(m + 1))
    for j in range(1, n + 1):
        ndp = [j]
        for i in range(1, m + 1):
            ndp.append(min(dp[i]+1, ndp[-1]+1, dp[i-1]+(0 if a[i-1]==b[j-1] else 1)))
        dp = ndp
    return 1 - dp[-1] / max(len(a), len(b))

aligned = []
used = set()
print("\nMatching:")
for ll in lrc_in_video:
    best_sc, best_ar = 0, None
    for ar in asr_results:
        if ar['seg'] in used: continue
        sc = lev_ratio(ll['text'], ar['text'])
        if sc > best_sc: best_sc = sc; best_ar = ar
    if best_ar and best_sc > 0.1:
        used.add(best_ar['seg'])
        aligned.append({'text': ll['text'], 'start': best_ar['start'], 'end': best_ar['end'], 'score': round(best_sc, 2)})
        print(f"  [sc={best_sc:.2f}] \"{ll['text'][:30]}\" -> [{best_ar['start']:.0f}-{best_ar['end']:.0f}s]")
    else:
        print(f"  [MISS] \"{ll['text'][:30]}\"")

print(f"\nAligned: {len(aligned)}/{len(lrc_in_video)}")

# 生成ASS
ass_path = os.path.join(OUTPUT_DIR, 'subtitle.ass')
with open(ass_path, 'w', encoding='utf-8-sig') as f:
    f.write('[Script Info]\nTitle: Lyrics\nScriptType: v4.00+\n')
    f.write('PlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n')
    f.write('[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n')
    f.write('Style: Default,Microsoft YaHei,45,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,3,1,2,80,80,40,1\n\n')
    f.write('[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n')
    def fmt(t):
        h=int(t//3600); m=int((t%3600)//60); s=t%60
        return f"{h}:{m:02d}:{s:05.2f}"
    for i, a in enumerate(aligned):
        end = aligned[i+1]['start'] if i+1 < len(aligned) else min(a['end']+3, duration)
        f.write(f"Dialogue: 0,{fmt(a['start'])},{fmt(end)},Default,,0,0,0,,{a['text']}\n")

# FFmpeg烧录
output_video = os.path.join(OUTPUT_DIR, 'huangtang_subtitled.mp4')
ass_esc = ass_path.replace('\\','/').replace(':','\\:')
print(f"\nBurning subtitles...")
r = subprocess.run(['ffmpeg','-y','-i', VIDEO_PATH, '-vf', f"ass='{ass_esc}'",
                    '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                    '-c:a','copy', output_video], capture_output=True)
if r.returncode != 0:
    print(f"FFmpeg ERROR: {r.stderr[-500:].decode('utf-8','replace')}")
else:
    size = os.path.getsize(output_video)/1024/1024
    print(f"[OK] {output_video} ({size:.1f}MB)")

print("\nALL DONE!")
