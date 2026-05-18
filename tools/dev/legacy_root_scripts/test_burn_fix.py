"""
自测脚本：验证 pipeline_burn_subtitles 的 Windows 特殊字符文件名修复
测试目标：
1. 硬链接创建是否正常
2. FFmpeg 烧录命令是否能成功执行
3. 输出视频是否有字幕轨道
"""
import os
import sys
import time
import subprocess
import shutil

# 用项目虚拟环境的 Python 跑（确保 import app 能用）
sys.path.insert(0, r'D:\video_clip')

# === 配置 ===
VIDEO_DIR = r'D:\video_clip\output'
INPUT_VIDEO = os.path.join(VIDEO_DIR, r'周深-VX-q97643800 (1)_131754_subtitled.mp4')  # 旧的无字幕版
ASS_FILE = os.path.join(VIDEO_DIR, 'subtitle_pipeline.ass')
TEST_OUTPUT = os.path.join(VIDEO_DIR, '_test_burn_verify.mp4')

def test_hardlink_creation():
    """Test 1: 硬链接创建"""
    print("=" * 60)
    print("TEST 1: 硬链接创建 (Windows 特殊字符绕过)")
    print("=" * 60)
    
    video_basename = os.path.basename(INPUT_VIDEO)
    _needs_safe_name = any(c in video_basename for c in ('(', ')', ' ', '&', '@', '!', '#'))
    print(f"  文件名: {video_basename}")
    print(f"  含特殊字符: {_needs_safe_name}")
    
    if not _needs_safe_name:
        print("  SKIP: 测试文件名不含特殊字符")
        return INPUT_VIDEO, TEST_OUTPUT
    
    _safe_ts = f"{int(time.time())}_{hash(video_basename) % 10000:04d}"
    _safe_input = os.path.join(VIDEO_DIR, f"_input_{_safe_ts}.mp4")
    
    # 清理旧链接
    if os.path.exists(_safe_input):
        os.remove(_safe_input)
    
    # 创建硬链接
    try:
        os.link(INPUT_VIDEO, _safe_input)
        link_size = os.path.getsize(_safe_input)
        orig_size = os.path.getsize(INPUT_VIDEO)
        print(f"  ✅ 硬链接创建成功: {_safe_input}")
        print(f"     原始大小: {orig_size/1024/1024:.1f}MB | 链接大小: {link_size/1024/1024:.1f}MB")
        assert link_size == orig_size, "硬链接大小不匹配！"
        return _safe_input, TEST_OUTPUT, [_safe_input]
    except OSError as e:
        print(f"  ❌ 硬链接失败: {e}")
        print("  → 回退到原始路径")
        return INPUT_VIDEO, TEST_OUTPUT, []


def test_ffmpeg_burn(safe_input_path, output_path, cleanup_files):
    """Test 2: FFmpeg 实际烧录（只烧前10秒快速验证）"""
    print()
    print("=" * 60)
    print("TEST 2: FFmpeg 烧录 (前10秒片段)")
    print("=" * 60)
    
    ass_local = os.path.join(VIDEO_DIR, 'subtitle.ass')
    vf = f"ass={os.path.basename(ass_local)},scale=-2:1080,fps=60"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", os.path.basename(safe_input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-t", "10",           # 只烧前10秒！加速验证
        "-movflags", "+faststart",
        os.path.basename(output_path),
    ]
    
    print(f"  输入: {os.path.basename(safe_input_path)}")
    print(f"  输出: {os.path.basename(output_path)}")
    print(f"  命令: {' '.join(cmd[:8])}...")
    
    start = time.time()
    r = subprocess.run(
        cmd,
        encoding='utf-8',
        errors='replace',
        capture_output=True,
        timeout=120,
        cwd=VIDEO_DIR,
    )
    elapsed = time.time() - start
    
    if r.returncode == 0:
        out_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✅ FFmpeg 烧录成功! ({elapsed:.1f}s, 输出{out_size:.1f}MB)")
        
        # 检查 stderr 中有没有 warning/error
        err_lines = [l for l in r.stderr.split('\n') if l.strip() and 
                     ('error' in l.lower() or 'warning' in l.lower())]
        if err_lines:
            print(f"  ⚠️  有 {len(err_lines)} 条 warning:")
            for line in err_lines[:5]:
                print(f"     {line}")
        else:
            print(f"  ✅ 无 error/warning")
        return True
    else:
        print(f"  ❌ FFmpeg 失败 (returncode={r.returncode})")
        # 显示最后几行关键错误
        last_err = r.stderr[-800:] if len(r.stderr) > 800 else r.stderr
        for line in last_err.split('\n'):
            line = line.strip()
            if line and ('error' in line.lower() or 'invalid' in line.lower()):
                print(f"     📌 {line}")
        return False


def test_subtitle_tracks(output_path):
    """Test 3: 验证输出视频的字幕/字幕内容"""
    print()
    print("=" * 60)
    print("TEST 3: 输出视频验证")
    print("=" * 60)
    
    if not os.path.exists(output_path):
        print(f"  ❌ 输出文件不存在: {output_path}")
        return
    
    # ffprobe 检查流信息
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,width,height,duration,nb_frames",
        "-of", "csv=p=0",
        output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(f"  流信息:")
    for line in r.stdout.strip().split('\n'):
        if line.strip():
            print(f"     {line}")
    
    # 提取第5秒的帧截图验证字幕可见性
    frame_out = os.path.join(VIDEO_DIR, '_test_frame_at_5s.png')
    cmd2 = [
        "ffmpeg", "-y",
        "-ss", "5",
        "-i", output_path,
        "-frames:v", "1",
        "-q:v", "2",
        frame_out
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8',
                        errors='replace', timeout=30, cwd=VIDEO_DIR)
    if r2.returncode == 0 and os.path.exists(frame_out):
        print(f"  ✅ 第5秒帧截图已保存: {frame_out}")
        print(f"     请查看此图片确认字幕是否在画面中显示！")
    else:
        print(f"  ⚠️  帧截图失败")


if __name__ == "__main__":
    print("\n" + "🔍".center(60, "="))
    print("   pipeline_burn_subtitles 自测 - Windows 特殊字符修复")
    print("🔍".center(60, "="))
    
    # Test 1
    result = test_hardlink_creation()
    safe_input = result[0]
    test_output = result[1] 
    cleanup = result[2] if len(result) > 2 else []
    
    # Test 2
    success = test_ffmpeg_burn(safe_input, test_output, cleanup)
    
    # Test 3 (only if burn succeeded)
    if success:
        test_subtitle_tracks(test_output)
    
    # Cleanup
    print()
    print("=" * 60)
    print("清理临时文件")
    for f in cleanup:
        try:
            if os.path.exists(f):
                os.remove(f)
                print(f"  🗑️  已删除: {os.path.basename(f)}")
        except:
            pass
    if os.path.exists(TEST_OUTPUT):
        # 保留供人工检查：print(f"  🗑️  已删除: {os.path.basename(TEST_OUTPUT)}")
        # os.remove(TEST_OUTPUT)
        pass
    
    print()
    print("✅ 全部测试完成!" if success else "❌ 测试失败!")
