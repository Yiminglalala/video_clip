"""
自测脚本：验证 F1-F5 字幕修复效果
测试视频：周深-VX-q97643800 (1).mp4 (157秒)
预期：字幕应均匀分布在整个视频时长内，不再出现"看不到字幕"的问题
"""
import sys
import os
import json
import time
import traceback

# 添加项目路径
sys.path.insert(0, r"D:\video_clip")

def main():
    print("=" * 60)
    print("  字幕修复 F1-F5 自测")
    print("=" * 60)
    
    test_video = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (1).mp4"
    output_dir = r"D:\video_clip\output\test_fix_" + time.strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(test_video):
        print(f"❌ 测试视频不存在: {test_video}")
        return False
    
    # Step 0: 导入 app 模块（会触发 Streamlit 初始化）
    print("\n📦 Step 0: 加载 app.py ...")
    
    # 需要模拟 Streamlit 的 session_state
    import streamlit as st
    
    # 初始化 session_state（如果还没初始化）
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = output_dir
    else:
        st.session_state.output_dir = output_dir
    
    # 设置日志回调
    log_lines = []
    def log_fn(msg):
        log_lines.append(msg)
        t = time.strftime("%H:%M:%S")
        print(f"  [{t}] {msg}")
    st.session_state._log_func = log_fn
    
    # 导入核心函数
    try:
        from src.processor import LiveVideoProcessor, ProcessingConfig
        from src.lyric_subtitle import fetch_lyrics, parse_lrc_text
        import numpy as np
        import soundfile as sf
        
        # 导入 app.py 的函数（需要避免重复导入问题）
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", r"D:\video_clip\app.py")
        app_module = importlib.util.module_from_spec(spec)
        
        # 设置必要的环境变量
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        
        # 手动加载（跳过 __main__ 入口）
        print("  ✅ 模块依赖加载成功")
    except Exception as e:
        print(f"  ❌ 模块加载失败: {e}")
        traceback.print_exc()
        return False
    
    # Step 1: 提取音频
    print(f"\n🎵 Step 1: 提取音频 from {os.path.basename(test_video)} ...")
    
    audio_path = os.path.join(output_dir, "test_audio.wav")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 用 ffmpeg 提取音频
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", test_video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            print(f"  ❌ FFmpeg 失败: {result.stderr[-300:].decode('utf-8', errors='replace')}")
            return False
        
        # 读取音频
        audio_16k, sr = sf.read(audio_path)
        duration_sec = len(audio_16k) / sr
        print(f"  ✅ 音频提取成功: {duration_sec:.1f}s, sr={sr}Hz")
    except Exception as e:
        print(f"  ❌ 音频提取失败: {e}")
        traceback.print_exc()
        return False
    
    # Step 2: 获取 LRC 歌词（用已知的歌曲名）
    print("\n📝 Step 2: 搜索歌词...")
    
    # 从之前的报告知道这首歌是《再见》by 周深
    song_title = "再见"
    song_artist = "周深"
    
    try:
        lrc_text = fetch_lyrics(song_title, song_artist)
        if lrc_text:
            lrc_lines = parse_lrc_text(lrc_text)
            print(f"  ✅ 获取到歌词: {len(lrc_lines)} 行 | 前3行: {[l['text'][:15] for l in lrc_lines[:3]]}")
        else:
            print("  ⚠️ 未搜到歌词，使用测试数据")
            # 使用之前 pipeline_report.json 中记录的 LRC 内容
            lrc_text = """[ar:周深]
[ti:再见]
[al:深空间]
[00:01.23]再見 我們要去明天
[00:05.67]再見 和你說聲再見
[00:09.89]我最親愛的朋友
[00:14.12]我最可愛的遇見
[00:18.45]我最羈絆的幾年
[00:22.78]未來也許不如願
[00:27.01]但我仍為你祝願
[00:31.34]願歲月不會改變你的臉
[00:35.67]回廊裡話語輕得像夢醒的哈欠
[00:40.00]原來在告別的季節 陽光會刺眼
[00:44.33]我不懷念 我只無言
[00:48.66]當鈴聲不再響 我竟忽然想
[00:52.99]課無妨再拖一堂
[00:57.32]不知 丟在哪的筆記 我也在那裡
[01:01.65]打鬧間那些不經意竟成為點滴
[01:05.98]我不明白 我會明白
[01:10.31]第一課才剛要開
[01:14.64]看 窗外的天再不是過去的答卷
[01:18.97]結伴的冒險 痛才熱烈 記得借你的那只筆
[01:23.30]換來成長一場苦戀 人生沒有荒廢可言
[01:27.63]說起明天勇氣不減 青春多像你我渴望翻越的牆沿
[01:31.96]再見 和你說聲再見
[01:36.29]我最親愛的朋友
[01:40.62]我最可愛的遇見
[01:44.95]我最羈絆的幾年
[01:49.28]再見 我們要去明天
[01:53.61]未來也許不如願
[01:57.94]但我仍為你祝願
[02:02.27]曾經說過的理想要實現
[02:06.60]這 世界很大 再遲疑來不及走遍
[02:10.93]這一程我將獨自體驗曾好奇的心
[02:15.26]請不要因為成長淪為經驗
[02:19.59]我依然還有那麼多勇氣走遠
[02:23.92]我也明白 世界是你我前後左右每一張臉
[02:28.25]怎麼 我竟忽然懷念
[02:32.58]連綿夏日的雨呀
[02:36.91]關於憧憬的談天
[02:41.24]我們多少徹夜不眠
[02:45.57]最後 我們就要告別
[02:49.90]告別我們曾經一字一點 一笑一言
[02:54.23]揮揮手再向前
[02:58.56]或許將來某一天 青春已遙遠
[03:03.00]玩笑話的未來 忽然間到來
[03:10.50]那麼我們再見
[03:18.00]別忘了曾去過彼此的世界
"""
            lrc_lines = parse_lrc_text(lrc_text)
            print(f"  ✅ 使用内置歌词: {len(lrc_lines)} 行")
            
    except Exception as e:
        print(f"  ❌ 歌词获取失败: {e}")
        traceback.print_exc()
        lrc_text = None
        lrc_lines = []
    
    # Step 3: 运行 ASR + 匹配 + 锚点插值
    print(f"\n🤖 Step 3: 运行 ASR (Whisper large-v3-turbo) + 匹配 + 插值 ...")
    
    try:
        from app import run_pipeline
        
        output_mp4 = os.path.join(output_dir, "fixed_subtitled.mp4")
        
        start_time = time.time()
        success, result = run_pipeline(
            test_video,
            audio_16k,
            sr,
            lrc_text or "",
            output_mp4,
            engine="fusion"
        )
        elapsed = time.time() - start_time
        
        if success:
            print(f"\n  ✅ Pipeline 成功! 耗时 {elapsed:.1f}s")
            
            # 读取生成的 report
            report_path = os.path.join(output_dir, "pipeline_report.json")
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                print(f"\n📊 Pipeline Report:")
                print(f"  引擎:     {report.get('engine', '?')}")
                print(f"  ASR词数:  {report.get('asr_word_count', 0)}")
                print(f"  LRC行数:  {report.get('lrc_line_count', 0)}")
                print(f"  匹配数:   {report.get('matched_count', 0)}")
                print(f"  插值数:   {report.get('interpolated_count', 0)}")
                print(f"  总字幕行: {report.get('total_subtitle_count', 0)}")
                
                # === 关键验证 ===
                matches = report.get('matches', [])
                words_preview = report.get('asr_words_preview', [])
                
                print(f"\n🔍 F1验证 - 垃圾词过滤:")
                long_words = [w for w in words_preview 
                             if w.get('end', 0) - w.get('start', 0) > 8]
                if long_words:
                    print(f"  ⚠️ 还有 {len(long_words)} 个超长词未过滤!")
                    for w in long_words[:3]:
                        dur = w.get('end', 0) - w.get('start', 0)
                        print(f"     '{w.get('text','')[:30]}' ({dur:.1f}s)")
                else:
                    print(f"  ✅ 无超长词 (>8s)! 过滤生效。共{len(words_preview)}个有效词")
                    # 显示前5个词
                    for w in words_preview[:5]:
                        dur = w.get('end', 0) - w.get('start', 0)
                        print(f"     '{w.get('text','')[:25]}' @ {w.get('start',0):.1f}-{w.get('end',0):.1f}s ({dur:.1f}s)")
                
                print(f"\n🔍 F3验证 - 字幕时间轴分布:")
                if len(matches) > 2:
                    first_start = matches[0].get('start_sec', 0)
                    last_end = matches[-1].get('end_sec', 0)
                    
                    # 检查前半段覆盖率
                    mid_point = last_end / 2
                    first_half_matches = [m for m in matches if m.get('start_sec', 0) < mid_point]
                    
                    print(f"  首字时间:   {first_start:.1f}s (理想 <15s)")
                    print(f"  末字时间:   {last_end:.1f}s (视频总长 ~{duration_sec:.0f}s)")
                    print(f"  前半段匹配: {len(first_half_matches)}/{len(matches)} 行 ({100*len(first_half_matches)/max(len(matches),1):.0f}%)")
                    
                    # 检查是否有挤压现象
                    squeeze_clusters = []
                    for i in range(min(10, len(matches)-1)):
                        span = matches[i+1]['start_sec'] - matches[i]['start_sec']
                        if span < 1.0:
                            squeeze_clusters.append(i)
                    
                    if squeeze_clusters:
                        print(f"  ⚠️ 有 {len(squeeze_clusters)} 行挤在<1s窗口")
                        for idx in squeeze_clusters[:3]:
                            m = matches[idx]
                            print(f"     '{m.get('lrc_text','')[:20]}' @ {m.get('start_sec',0):.1f}s")
                    else:
                        print(f"  ✅ 无明显挤压现象!")
                    
                    # 显示前8行和后3行的时间分布
                    print(f"\n  前8行时间分布:")
                    for m in matches[:8]:
                        status = "✓确认" if not m.get('interpolated') else "~插值"
                        print(f"     [{status}] {m.get('start_sec',0):7.1f}s - {m.get('end_sec',0):6.1f}s | '{m.get('lrc_text','')[:22]}'")
                    
                    print(f"  后3行时间分布:")
                    for m in matches[-3:]:
                        status = "✓确认" if not m.get('interpolated') else "~插值"
                        print(f"     [{status}] {m.get('start_sec',0):7.1f}s - {m.get('end_sec',0):6.1f}s | '{m.get('lrc_text','')[:22]}'")
                    
                    # 最终判定
                    all_ok = True
                    issues = []
                    
                    if first_start > 20:
                        all_ok = False
                        issues.append(f"首字过晚({first_start:.1f}s)")
                    if len(squeeze_clusters) > 5:
                        all_ok = False
                        issues.append(f"过多挤压行({len(squeeze_clusters)})")
                    if len(first_half_matches) / max(len(matches), 1) < 0.3:
                        all_ok = False
                        issues.append(f"前半段覆盖率过低({100*len(first_half_matches)/max(len(matches),1):.0f}%)")
                    if long_words:
                        all_ok = False
                        issues.append(f"仍有{len(long_words)}个超长词")
                    
                    print(f"\n{'=' * 40}")
                    if all_ok:
                        print(f"  ✅ 全部验证通过! 字幕修复成功!")
                    else:
                        print(f"  ⚠️ 存在问题: {'; '.join(issues)}")
                    print(f"{'=' * 40}")
                    
                else:
                    print(f"  ⚠️ 匹配结果太少，无法验证时间轴分布")
            
            # 验证输出文件
            if os.path.exists(output_mp4):
                size_mb = os.path.getsize(output_mp4) / (1024 * 1024)
                print(f"\n🎬 输出文件: {output_mp4} ({size_mb:.1f} MB)")
            else:
                print(f"\n⚠️ 输出文件不存在: {output_mp4}")
            
        else:
            print(f"\n  ❌ Pipeline 失败: {result}")
            
    except Exception as e:
        print(f"\n  ❌ Pipeline 执行异常: {e}")
        traceback.print_exc()
        return False
    
    print(f"\n{'=' * 60}")
    print(f"  自测完成! 日志输出在: {output_dir}")
    print(f"{'=' * 60}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
