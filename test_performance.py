# -*- coding: utf-8 -*-
"""
系统性能测试脚本
- 测试多种视频样本
- 验证匹配成功率
- 检查字幕质量
- 生成性能报告
"""
import sys, os, time, json
import numpy as np
import soundfile as sf
from datetime import datetime

# 项目根目录
sys.path.insert(0, r"D:\video_clip")
os.chdir(r"D:\video_clip")

# 模拟 session_state 避免 Streamlit 报错
class MockState:
    output_dir = r"D:\video_clip\output"
    whisper_model_name = "medium"
    def __getattr__(self, name):
        if name == "whisper_model_name":
            return "medium"
        return None

# 模拟 Streamlit 模块
class MockStreamlit:
    class MockSpinner:
        def __init__(self, text):
            print(f"[Mock] {text}")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    spinner = MockSpinner
    session_state = None

import sys
class MockModule:
    pass

# 替换 streamlit 模块
sys.modules['streamlit'] = MockModule()
sys.modules['streamlit'].st = MockStreamlit()
sys.modules['streamlit'].session_state = MockState()

# 现在导入 app
import os
os.environ['STREAMLIT_RUNTIME'] = 'test'

from importlib import import_module
app = import_module('app')

# 设置 session_state
app.st.session_state = MockState()
app.st.session_state.output_dir = r"D:\video_clip\output"
app.st.session_state.whisper_model_name = "medium"

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
    },
    {
        "name": "测试样本2 - 简单歌词",
        "audio_path": r"D:\video_clip\A-Lin_2021_concert.wav",
        "lrc_text": """[ti:简单测试]
[ar:测试歌手]
[00:05.00]你好世界
[00:10.00]这是测试
[00:15.00]简单歌词
[00:20.00]用于测试""",
        "expected_lines": 4
    }
]

def mock_log(msg):
    """模拟日志输出"""
    print(f"  {msg}")

def test_asr_fusion(audio_path, lrc_text, sample_name):
    """测试双引擎融合性能"""
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
    
    # 测试融合函数
    t0 = time.time()
    try:
        result = app.run_asr_fusion_v2(audio, sr, lrc_text, log_fn=mock_log)
        elapsed = round(time.time() - t0, 1)
    except Exception as e:
        print(f"❌ 融合测试失败: {e}")
        return None
    
    if not result:
        print("❌ 融合结果为空")
        return None
    
    # 分析结果
    analysis = {
        "sample_name": sample_name,
        "duration": len(audio)/sr,
        "elapsed_time": elapsed,
        "engine": result.get("engine", "未知"),
        "word_source_engine": result.get("word_source_engine", "未知"),
        "word_count": len(result.get("words", [])),
        "match_count": len(result.get("matches", [])),
        "alignment_meta": result.get("alignment_meta", {}),
        "fusion_log": result.get("fusion_log", ""),
        "scores": result.get("scores", {})
    }
    
    # 计算匹配成功率
    lrc_lines = app.pipeline_parse_lrc(lrc_text)
    expected_lines = len(lrc_lines)
    actual_matches = len(result.get("matches", []))
    match_rate = (actual_matches / expected_lines * 100) if expected_lines > 0 else 0
    analysis["match_rate"] = match_rate
    analysis["expected_lines"] = expected_lines
    analysis["actual_matches"] = actual_matches
    
    # 输出分析结果
    print(f"\n{'='*70}")
    print(f"测试结果分析")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed}s")
    print(f"胜出引擎: {analysis['engine']}")
    print(f"词源引擎: {analysis['word_source_engine']}")
    print(f"词数: {analysis['word_count']}")
    print(f"预期行数: {expected_lines}")
    print(f"实际匹配: {actual_matches}")
    print(f"匹配率: {match_rate:.1f}%")
    
    if result.get("alignment_meta"):
        meta = result["alignment_meta"]
        print(f"\n对齐元数据:")
        print(f"  确认率: {meta.get('confirmed_ratio', 0):.1f}%")
        print(f"  需审核: {meta.get('review_count', 0)}")
        print(f"  全局偏移: {meta.get('global_offset', 0):.2f}s")
        print(f"   tempo缩放: {meta.get('tempo_scale', 1):.3f}")
        print(f"  漂移风险: {meta.get('drift_risk', '未知')}")
    
    if result.get("scores"):
        print(f"\n引擎评分:")
        for name, info in result["scores"].items():
            print(f"  {name}:")
            print(f"    词数: {info.get('word_count', 0)}")
            print(f"    确认率: {info.get('confirmed_ratio', 0):.3f}")
            print(f"    错误: {info.get('error', '无')}")
    
    print(f"\n融合日志: {result.get('fusion_log', '无')}")
    
    return analysis

def main():
    """主测试函数"""
    print("="*80)
    print("🎤 演唱会字幕生成器性能测试")
    print("="*80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # 测试每个样本
    for sample in test_samples:
        result = test_asr_fusion(
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
        report_path = f"output/test_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
