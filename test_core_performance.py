# -*- coding: utf-8 -*-
"""
核心性能测试脚本
- 直接测试核心功能
- 避免Streamlit依赖
"""
import sys, os, time, json
import numpy as np
import soundfile as sf
from datetime import datetime

# 项目根目录
sys.path.insert(0, r"D:\video_clip")
os.chdir(r"D:\video_clip")

# 直接导入需要的模块
from src.alignment_engine import (
    align_lrc_monotonic,
    build_hotword_text,
    compute_asr_quality,
    fuse_engine_alignments,
)

# 导入音频处理函数
def preprocess_audio(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """
    音频预处理流水线（演唱会人声增强版）
    """
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # 带通滤波 100-4000Hz
    import scipy.signal as signal
    sos_bp = signal.butter(4, [100.0, 4000.0], btype='band', fs=sr, output='sos')
    filtered = signal.sosfilt(sos_bp, audio_data)
    
    # 预加重
    filtered = np.append(filtered[0], filtered[1:] - 0.97 * filtered[:-1])
    
    # 动态压缩
    threshold = 0.15 * np.max(np.abs(filtered))
    if threshold > 0:
        abs_sig = np.abs(filtered)
        mask = abs_sig > threshold
        compressed = filtered.copy()
        compressed[mask] = (
            np.sign(filtered[mask]) *
            (threshold + (abs_sig[mask] - threshold) * 
             np.tanh((abs_sig[mask] - threshold) / (threshold + 1e-8)))
        )
        filtered = compressed
    
    # 峰值归一化到 0.9
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        normalized = filtered * (0.9 / max_val)
    else:
        normalized = filtered
    
    return normalized.astype(np.float32)

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def pipeline_parse_lrc(lrc_text):
    """
    解析 LRC 文本为有序时间轴
    """
    import re
    def _parse_time(ts_str):
        m, s = ts_str.split(':')
        return int(m) * 60 + float(s)

    lines = []
    has_timestamps = False
    CREDIT_KEYWORDS = ['作词', '作曲', '编曲', '混音', '监制', '制作人', '和声', '录音',
                       'Lyricist', 'Composer', 'Arranger', 'Producer', 'Mixing']
    for raw_line in lrc_text.strip().split('\n'):
        timestamps = re.findall(r'\[(\d{2}:\d{2}\.\d{2})\]', raw_line)
        text = re.sub(r'\[\d{2}:\d{2}\.\d{2}\]', '', raw_line).strip()
        if not text or text.startswith(('[ti:', '[ar:', '[al:', '[au:', '[by:', '[offset:', '[length:', '[tool:', '[ve:')):
            continue
        if any(kw in text for kw in CREDIT_KEYWORDS):
            continue
        if text and timestamps:
            has_timestamps = True
            lines.append({
                'time': _parse_time(min(timestamps)),
                'text': text,
            })
        elif text and not has_timestamps:
            lines.append({
                'time': None,
                'text': text,
            })

    if has_timestamps:
        lines = sorted(lines, key=lambda x: x['time'])

    return lines

def mock_log(msg):
    """模拟日志输出"""
    print(f"  {msg}")

# 测试样本配置
test_samples = [
    {
        "name": "测试样本1 - 不可一世",
        "audio_path": r"D:\video_clip\A-Lin_2021_concert.wav",
        "lrc_text": """[ti:不可一世]
[ar:谢霆锋]
[00:18.50]我害怕我不忍心再说
[00:23.20]一句我跟你
[00:27.80]终于让我看穿了爱情
[00:33.10]我明白这种游戏
[00:37.90]关于你不下剧演
[00:42.60]终于我莫承认了我伤心
[00:47.30]我确定不准回忆
[00:52.40]就是你的荒藏从口里去
[00:57.80]终于让我看穿了爱情""",
        "expected_lines": 9
    }
]

def test_core_performance(audio_path, lrc_text, sample_name):
    """
    测试核心性能
    """
    print(f"\n{'='*70}")
    print(f"[测试] {sample_name}")
    print(f"{'='*70}")
    
    # 加载音频
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # 转单声道
        print(f"音频: {len(audio)/sr:.1f}s | sr={sr}")
    except Exception as e:
        print(f"❌ 加载音频失败: {e}")
        return None
    
    # 预处理音频
    proc_audio = preprocess_audio(audio, sr)
    
    # 解析LRC
    lrc_lines = pipeline_parse_lrc(lrc_text)
    print(f"LRC歌词: {len(lrc_lines)} 行")
    
    # 测试对齐性能
    t0 = time.time()
    
    # 模拟ASR结果（使用简化版本）
    mock_asr_words = []
    for i, line in enumerate(lrc_lines):
        if line['time']:
            start = line['time']
            end = start + 3.0  # 假设每行3秒
            mock_asr_words.append({
                "word": line['text'],
                "start": start,
                "end": end
            })
    
    # 测试对齐函数
    try:
        alignment = align_lrc_monotonic(
            asr_words=mock_asr_words,
            lrc_lines=lrc_lines,
            strictness="standard",
            low_conf_threshold=0.25,
            max_drift_sec=2.5,
            engine_name="mock"
        )
        elapsed = round(time.time() - t0, 1)
    except Exception as e:
        print(f"❌ 对齐测试失败: {e}")
        return None
    
    # 分析结果
    analysis = {
        "sample_name": sample_name,
        "duration": len(audio)/sr,
        "elapsed_time": elapsed,
        "alignment_meta": alignment.get("alignment_meta", {}),
        "match_count": len(alignment.get("confirmed_matches", [])),
        "expected_lines": len(lrc_lines)
    }
    
    # 计算匹配率
    expected_lines = len(lrc_lines)
    actual_matches = len(alignment.get("confirmed_matches", []))
    match_rate = (actual_matches / expected_lines * 100) if expected_lines > 0 else 0
    analysis["match_rate"] = match_rate
    
    # 输出分析结果
    print(f"\n{'='*70}")
    print(f"测试结果分析")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed}s")
    print(f"预期行数: {expected_lines}")
    print(f"实际匹配: {actual_matches}")
    print(f"匹配率: {match_rate:.1f}%")
    
    if alignment.get("alignment_meta"):
        meta = alignment["alignment_meta"]
        print(f"\n对齐元数据:")
        print(f"  确认率: {meta.get('confirmed_ratio', 0):.1f}%")
        print(f"  需审核: {meta.get('review_count', 0)}")
        print(f"  全局偏移: {meta.get('global_offset', 0):.2f}s")
        print(f"   tempo缩放: {meta.get('tempo_scale', 1):.3f}")
        print(f"  漂移风险: {meta.get('drift_risk', '未知')}")
    
    return analysis

def main():
    """主测试函数"""
    print("="*80)
    print("🎤 演唱会字幕生成器核心性能测试")
    print("="*80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # 测试每个样本
    for sample in test_samples:
        result = test_core_performance(
            sample["audio_path"],
            sample["lrc_text"],
            sample["name"]
        )
        if result:
            test_results.append(result)
    
    # 生成测试报告
    if test_results:
        report = {
            "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_samples": len(test_results),
            "average_match_rate": sum(r["match_rate"] for r in test_results) / len(test_results),
            "average_time": sum(r["elapsed_time"] for r in test_results) / len(test_results),
            "details": test_results
        }
        
        # 保存报告
        report_path = f"output/test_core_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print("测试报告摘要")
        print(f"{'='*80}")
        print(f"总样本数: {report['total_samples']}")
        print(f"平均匹配率: {report['average_match_rate']:.1f}%")
        print(f"平均处理时间: {report['average_time']:.1f}s")
        print(f"报告已保存: {report_path}")
    else:
        print("\n❌ 所有测试失败")

if __name__ == "__main__":
    main()
