#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时测试脚本：使用现有字幕文件烧录到视频
"""

import os
import sys
import subprocess
import shutil
import tempfile

def burn_subtitle_with_larger_font(video_path, subtitle_path, output_path, font_size=120):
    """
    使用更大的字体大小烧录字幕
    
    Args:
        video_path: 输入视频路径
        subtitle_path: 输入字幕文件路径
        output_path: 输出视频路径
        font_size: 字体大小，默认120
    """
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return False
    
    if not os.path.exists(subtitle_path):
        print(f"错误：字幕文件不存在: {subtitle_path}")
        return False
    
    # 复制并修改字幕文件，增大字体大小
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        ass_content = f.read()
    
    # 修改字体大小
    import re
    # 查找并替换字体大小
    ass_content = re.sub(r'Fontsize:\s*\d+', f'Fontsize: {font_size}', ass_content)
    
    # 创建临时字幕文件
    temp_ass = tempfile.mktemp(suffix='.ass')
    with open(temp_ass, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    print(f"已创建临时字幕文件，字体大小: {font_size}")
    
    # 检测视频方向
    def detect_orientation(video_path):
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            if result.returncode != 0:
                return "landscape"
            
            import json
            info = json.loads(result.stdout)
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    width = int(stream.get('width', 0))
                    height = int(stream.get('height', 0))
                    if width > height:
                        return "landscape"
                    else:
                        return "portrait"
        except Exception as e:
            print(f"检测视频方向时出错: {e}")
        return "landscape"
    
    orientation = detect_orientation(video_path)
    print(f"视频方向: {orientation}")
    
    # 根据视频方向设置目标分辨率
    if orientation == "portrait":
        target_w, target_h = 1080, 1920
        target_dar = "9/16"
    else:
        target_w, target_h = 1920, 1080
        target_dar = "16/9"
    
    # FFmpeg 烧录字幕
    video_dir = os.path.dirname(video_path)
    ass_local = os.path.join(video_dir, os.path.basename(temp_ass))
    same_ass_path = os.path.abspath(ass_local) == os.path.abspath(temp_ass)
    if not same_ass_path:
        shutil.copy2(temp_ass, ass_local)
    
    # 视频滤镜
    video_filter = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={target_w}:{target_h},"
        f"format=yuv420p,"
        f"ass={os.path.basename(ass_local)},"
        f"setsar=1,setdar={target_dar}"
    )
    
    # 构建FFmpeg命令（优先使用GPU加速）
    nvenc_cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", video_filter,
        "-c:v", "h264_nvenc",
        "-rc", "cbr", "-b:v", "8M", "-maxrate", "8M", "-bufsize", "16M", "-preset", "p4",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-pix_fmt", "yuv420p",
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
        "-pix_fmt", "yuv420p",
        "-r", "60",
        "-movflags", "+faststart",
        output_path,
    ]
    
    print(f"开始烧录字幕到视频...")
    print(f"输出文件: {output_path}")
    
    try:
        print("尝试使用GPU加速...")
        result = subprocess.run(nvenc_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=300, cwd=video_dir)
        
        if result.returncode != 0:
            if "h264_nvenc" in (result.stderr or ""):
                print("GPU加速失败，尝试使用CPU...")
                result = subprocess.run(cpu_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=300, cwd=video_dir)
            
            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "未知错误"
                print(f"FFmpeg 错误:\n{error_msg}")
                return False
        
        print("烧录完成！")
        return True
    except subprocess.TimeoutExpired:
        print("FFmpeg 编码超时（>5分钟）")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_ass):
            os.remove(temp_ass)
        if os.path.exists(ass_local) and (os.path.abspath(ass_local) != os.path.abspath(temp_ass)):
            os.remove(ass_local)

if __name__ == "__main__":
    # 测试参数
    video_path = "D:\\个人资料\\音乐测试\\视频素材\\live\\周深\\周深-VX-q97643800 (24).mp4"
    subtitle_path = "D:\\个人资料\\音乐测试\\视频素材\\live\\周深\\subtitle.ass"
    output_path = "D:\\video_clip\\output\\test_large_font.mp4"
    font_size = 90
    
    # 运行测试
    success = burn_subtitle_with_larger_font(video_path, subtitle_path, output_path, font_size)
    if success:
        print(f"\n测试完成！请查看输出文件: {output_path}")
    else:
        print("\n测试失败！")
