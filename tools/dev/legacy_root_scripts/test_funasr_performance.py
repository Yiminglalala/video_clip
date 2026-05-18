# -*- coding: utf-8 -*-
"""
FunASR引擎性能测试
- 测试FunASR的识别性能
- 验证匹配成功率
- 检查字幕质量
"""
import sys, os, time, json
import numpy as np
import soundfile as sf
from datetime import datetime

# 项目根目录
sys.path.insert(0, r"D:\video_clip")
os.chdir(r"D:\video_clip")

# 导入必要的模块
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

def run_funasr(audio: np.ndarray, sr: int, hotword: str = "") -> dict:
    """
    FunASR Paraformer - 词级时间戳，中文歌曲最准
    """
    import soundfile as sf
    import tempfile
    import torch
    
    # 加载FunASR模型
    from funasr import AutoModel
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc-c",
        spk_model="cam++",
    )
    
    # 写临时WAV文件
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=r"D:\video_clip\output")
    os.close(fd)
    sf.write(tmp_path, audio if sr == 16000 else resample_audio(audio, sr, 16000), 16000)
    
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    try:
        res = model.generate(
            input=tmp_path,
            batch_size_s=300,
            hotword=(hotword or "").strip()[:512],
        )
    finally:
        os.remove(tmp_path)  # 清理临时文件
    
    infer_time = time.time() - t0
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    
    words = []
    for seg in res[0]["sentence_info"]:
        text = seg.get('text', '').strip()
        if not text:
            continue
        words.append({
            "word": text,
            "start": float(seg['start']) / 1000.0,  # ms -> s
            "end": float(seg['end']) / 1000.0,
        })
    
    sentences = []
    current_sent = {"text": "", "start": 0, "end": 0}
    for w in words:
        if not current_sent["text"]:
            current_sent["text"] = w["word"]
            current_sent["start"] = w["start"]
            current_sent["end"] = w["end"]
        elif w["start"] - current_sent["end"] > 1.5:  # 1.5s gap = new sentence
            sentences.append(current_sent.copy())
            current_sent = {"text": w["word"], "start": w["start"], "end": w["end"]}
        else:
            current_sent["text"] += w["word"]
            current_sent["end"] = w["end"]
    if current_sent["text"]:
        sentences.append(current_sent)
    
    return {
        "engine": "funasr-paraformer",
        "text": "".join(w["word"] for w in words),
        "words": words,
        "sentences": sentences,
        "language": "zh",
        "metrics": {
            "infer_time": round(infer_time, 2),
            "rtf": round(infer_time / (len(audio)/sr), 3),
            "vram_peak_gb": round(vram_peak, 1),
        },
    }

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

def test_funasr_performance(audio_path, lrc_text, sample_name):
    """
    测试FunASR引擎性能
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
    
    # 测试FunASR性能
    t0 = time.time()
    try:
        funasr_result = run_funasr(proc_audio, sr)
        asr_time = funasr_result["metrics"]["infer_time"]
    except Exception as e:
        print(f"❌ FunASR测试失败: {e}")
        return None
    
    # 测试对齐性能
    t1 = time.time()
    try:
        alignment = align_lrc_monotonic(
            asr_words=funasr_result["words"],
            lrc_lines=lrc_lines,
            strictness="standard",
            low_conf_threshold=0.25,
            max_drift_sec=2.5,
            engine_name="funasr"
        )
        align_time = round(time.time() - t1, 1)
    except Exception as e:
        print(f"❌ 对齐测试失败: {e}")
        return None
    
    total_time = round(time.time() - t0, 1)
    
    # 分析结果
    analysis = {
        "sample_name": sample_name,
        "duration": len(audio)/sr,
        "total_time": total_time,
        "asr_time": asr_time,
        "align_time": align_time,
        "word_count": len(funasr_result["words"]),
        "alignment_meta": alignment.get("alignment_meta", {}),
        "match_count": len(alignment.get("confirmed_matches", [])),
        "expected_lines": len(lrc_lines),
        "asr_metrics": funasr_result.get("metrics", {})
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
    print(f"总耗时: {total_time}s")
    print(f"ASR时间: {asr_time}s")
    print(f"对齐时间: {align_time}s")
    print(f"词数: {analysis['word_count']}")
    print(f"预期行数: {expected_lines}")
    print(f"实际匹配: {actual_matches}")
    print(f"匹配率: {match_rate:.1f}%")
    
    if funasr_result.get("metrics"):
        metrics = funasr_result["metrics"]
        print(f"\nASR指标:")
        print(f"  推理时间: {metrics.get('infer_time', 0):.2f}s")
        print(f"  实时因子: {metrics.get('rtf', 0):.3f}")
        print(f"  显存峰值: {metrics.get('vram_peak_gb', 0):.1f}GB")
    
    if alignment.get("alignment_meta"):
        meta = alignment["alignment_meta"]
        print(f"\n对齐元数据:")
        print(f"  确认率: {meta.get('confirmed_ratio', 0):.1f}%")
        print(f"  需审核: {meta.get('review_count', 0)}")
        print(f"  全局偏移: {meta.get('global_offset', 0):.2f}s")
        print(f"   tempo缩放: {meta.get('tempo_scale', 1):.3f}")
        print(f"  漂移风险: {meta.get('drift_risk', '未知')}")
    
    # 打印前5个匹配结果
    matches = alignment.get("confirmed_matches", [])
    if matches:
        print(f"\n前5个匹配结果:")
        for i, m in enumerate(matches[:5]):
            print(f"  [{m['start_sec']:.1f}s-{m['end_sec']:.1f}s] {m['lrc_text']}")
    
    return analysis

def main():
    """主测试函数"""
    print("="*80)
    print("🎤 FunASR引擎性能测试")
    print("="*80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # 测试每个样本
    for sample in test_samples:
        result = test_funasr_performance(
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
            "average_total_time": sum(r["total_time"] for r in test_results) / len(test_results),
            "average_asr_time": sum(r["asr_time"] for r in test_results) / len(test_results),
            "average_align_time": sum(r["align_time"] for r in test_results) / len(test_results),
            "details": test_results
        }
        
        # 保存报告
        report_path = f"output/test_funasr_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print("测试报告摘要")
        print(f"{'='*80}")
        print(f"总样本数: {report['total_samples']}")
        print(f"平均匹配率: {report['average_match_rate']:.1f}%")
        print(f"平均总时间: {report['average_total_time']:.1f}s")
        print(f"平均ASR时间: {report['average_asr_time']:.1f}s")
        print(f"平均对齐时间: {report['average_align_time']:.1f}s")
        print(f"报告已保存: {report_path}")
    else:
        print("\n❌ 所有测试失败")

if __name__ == "__main__":
    main()
