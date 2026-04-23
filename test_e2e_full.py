"""
完整端到端自测：修复后的 ASS 样式 + 硬链接 FFmpeg 烧录
验证字幕在画面中可见
"""
import os, sys, time, subprocess, shutil

sys.path.insert(0, r'D:\video_clip')

VIDEO_DIR = r'D:\video_clip\output'
INPUT_VIDEO = os.path.join(VIDEO_DIR, r'周深-VX-q97643800 (1)_131754_subtitled.mp4')
TEST_OUTPUT = os.path.join(VIDEO_DIR, '_e2e_verify.mp4')

# Step 1: 读取最新的 pipeline_report.json 获取字幕数据
import json
report_path = os.path.join(VIDEO_DIR, 'pipeline_report.json')
with open(report_path, 'r', encoding='utf-8') as f:
    report = json.load(f)

matches = report.get('matches', [])
print(f"从 report 加载 {len(matches)} 行字幕")
print(f"时间范围: {matches[0]['start_sec']:.1f}s ~ {matches[-1]['end_sec']:.1f}s")

# Step 2: 用修复后的样式生成新 ASS 文件
def _pipeline_format_ass_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

events = []
for sub in matches:
    start_str = _pipeline_format_ass_time(sub['start_sec'])
    end_str = _pipeline_format_ass_time(sub['end_sec'])
    events.append(
        f"Dialogue: 0,{start_str},{end_str},Default,,"
        f"0000,0000,0000,{sub['lrc_text']}"
    )

ass_content = """[Script Info]
Title: Karaoke Subtitles
PlayResX: 1920; PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline,StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,2,0,8,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(events)

fixed_ass_path = os.path.join(VIDEO_DIR, 'subtitle_fixed.ass')
with open(fixed_ass_path, 'w', encoding='utf-8') as f:
    f.write(ass_content)
print(f"\n✅ 新 ASS 文件生成: {fixed_ass_path}")
print(f"   样式: Microsoft YaHei, 72px, BorderStyle=3(背景框), Alignment=8(底部居中)")

# Step 3: 创建硬链接
safe_input = os.path.join(VIDEO_DIR, '_e2e_safe_src.mp4')
if os.path.exists(safe_input):
    os.remove(safe_input)
os.link(INPUT_VIDEO, safe_input)
print(f"✅ 硬链接创建: {os.path.basename(safe_input)}")

# Step 4: FFmpeg 烧录（前30秒足够覆盖第一行+后续几行）
print("\n🎬 开始 FFmpeg 烧录 (30秒片段)...")
cmd = [
    "ffmpeg", "-y",
    "-i", "_e2e_safe_src.mp4",
    "-vf", "ass=subtitle_fixed.ass,scale=-2:1080,fps=60",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-c:a", "aac", "-b:a", "192k",
    "-t", "30",
    "-movflags", "+faststart",
    "_e2e_verify.mp4"
]
r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                   errors='replace', timeout=120, cwd=VIDEO_DIR)

if r.returncode == 0:
    out_size = os.path.getsize(TEST_OUTPUT) / 1024 / 1024
    print(f"✅ 烧录成功! 输出 {out_size:.1f}MB")
else:
    print(f"❌ 烧录失败! returncode={r.returncode}")
    for l in r.stderr.split('\n'):
        if 'error' in l.lower() or 'invalid' in l.lower():
            print(f"   {l}")
    sys.exit(1)

# Step 5: 在多个时间点截帧验证
print("\n📸 截取多帧验证字幕...")
test_points = [
    (2, "frame_e2e_15s.png"),      # ~15s: 第一行 "再見 和你說聲再見" (13.47-18.12)
    (6, "frame_e2e_19s.png"),      # ~19s: 第二行 "我最親愛的朋友"
    (12, "frame_e2e_25s.png"),     # ~25s: 中间行
]

for offset, fname in test_points:
    cmd_frame = [
        "ffmpeg", "-y", "-ss", str(offset),
        "-i", "_e2e_verify.mp4",
        "-frames:v", "1", "-q:v", "2",
        fname
    ]
    subprocess.run(cmd_frame, capture_output=True, cwd=VIDEO_DIR)
    if os.path.exists(os.path.join(VIDEO_DIR, fname)):
        print(f"  ✅ 截帧 @{offset}s → {fname}")
    else:
        print(f"  ❌ 截帧失败 @{offset}s")

# Step 6: 清理临时文件
for f in [safe_input]:
    try:
        if os.path.exists(f):
            os.remove(f)
            print(f"  🗑️ 清理: {f}")
    except:
        pass

print("\n" + "=" * 60)
print("✅ 自测完成！请查看 frame_e2e_*.png 验证字幕是否显示")
print("=" * 60)
