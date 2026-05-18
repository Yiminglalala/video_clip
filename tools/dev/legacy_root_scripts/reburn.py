"""用最新的v4 ASS重新烧录视频"""
import shutil, os, subprocess

VIDEO = r"D:\video_clip\output\A-Lin_声音梦境线上音乐会_20260412_1157\Song_01\Song_01_主歌_00m15s-00m59s.mp4"
ASS = r"D:\video_clip\output\lyric_aligned\subtitle_v4.ass"
OUTPUT = r"D:\video_clip\output\lyric_aligned\huangtang_v5_clean.mp4"

video_dir = os.path.dirname(VIDEO)
ass_copy = os.path.join(video_dir, 'subtitle.ass')
shutil.copy2(ASS, ass_copy)

video_name = os.path.basename(VIDEO)
ass_name = os.path.basename(ass_copy)

# 用 cwd 切换目录 + 纯文件名（避免路径冒号被解析为尺寸）
cmd = [
    "ffmpeg", "-y",
    "-i", video_name,
    "-vf", f"ass={ass_name}",
    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
    "-c:a", "copy",
    OUTPUT
]

print(f"Running FFmpeg in {video_dir}...")
r = subprocess.run(cmd, encoding='utf-8', errors='replace',
                   capture_output=True, timeout=120, cwd=video_dir)  # ✅ 用cwd切换目录
if r.returncode == 0:
    sz = os.path.getsize(OUTPUT) / (1024*1024)
    print(f"DONE: {OUTPUT} ({sz:.1f}MB)")
else:
    print(f"FAILED: {r.stderr[-300:]}")
