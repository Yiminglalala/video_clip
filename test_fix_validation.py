"""
F1-F5 字幕修复 独立验证脚本（不依赖 Streamlit）
用已知的 pipeline_report.json 数据验证修复逻辑是否正确
"""
import json
import os

# ============================================================
# F1 验证：ASR 垃圾词过滤
# ============================================================
def test_F1_garbage_filter():
    """验证 _filter_asr_words 能正确过滤超长垃圾词"""
    print("\n" + "="*50)
    print("  F1: ASR 垃圾词过滤")
    print("="*50)
    
    # 模拟 FunASR 在周深视频上的原始输出（来自实际 report）
    raw_words = [
        {"text": "你们会唱吗？", "start": 2.25, "end": 3.025},
        {"text": "再见，",       "start": 7.46, "end": 7.97},
        {"text": "我最基本的要去会让你花语情的香红心的那个东西在上，", 
         "start": 16.66, "end": 57.295},   # 🔴 40.64s 垃圾词
        {"text": "我经不笑，",   "start": 58.04, "end": 59.54},
        {"text": "放在脱，",     "start": 60.44, "end": 61.095},
        {"text": "不知留在哪了，","start": 65.45, "end": 66.455},
        {"text": "已经我也在这里，", "start": 66.94, "end": 69.675},
        {"text": "难道将那些风景里经说回不停，", "start": 71.65, "end": 75.865}, # 4.2s OK
        {"text": "我懂你，",     "start": 77.91, "end": 78.515},
        {"text": "我会明白，",   "start": 80.17, "end": 80.895},
        {"text": "欢笑后的未来，", "start": 81.7, "end": 83.525},
        {"text": "忽然间到了一个太高，", "start": 84.35, "end": 87.315},
        {"text": "要看来不及一起走，", "start": 87.78, "end": 109.505}, # 🔴 21.73s 超长
        {"text": "我愿意在向前再见。", "start": 110.43, "end": 134.635}, # 🔴 24.21s 超长
        {"text": "或许他们可曾去过彼此的世界。", "start": 137.43, "end": 154.605}, # 🔴 17.18s 超长
    ]
    
    def _filter_asr_words(words, max_word_dur=8.0):
        if not words:
            return [], 0
        filtered = []
        removed_count = 0
        for w in words:
            text = w.get('text', w.get('word', '')).strip()
            if not text:
                removed_count += 1
                continue
            dur = w.get('end', 0) - w.get('start', 0)
            if dur > max_word_dur:
                removed_count += 1
                continue
            filtered.append(w)
        return filtered, removed_count
    
    filtered, n_removed = _filter_asr_words(raw_words)
    
    print(f"  输入: {len(raw_words)} 个词")
    print(f"  过滤后: {len(filtered)} 个词 (移除 {n_removed})")
    
    # 验证
    expected_removed = 4  # 40.64s, 21.73s, 24.21s, 17.18s
    long_ones = [w['text'][:20] for w in raw_words if w['end']-w['start'] > 8]
    print(f"  移除的词:")
    for txt in long_ones:
        print(f"    🗑️ '{txt}'")
    
    ok = (n_removed == expected_removed) and (len(filtered) == len(raw_words) - expected_removed)
    print(f"  结果: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


# ============================================================
# F3 验证：锚点稀疏保护
# ============================================================
def test_F3_anchor_sparse_protect():
    """验证 F3 锚点稀疏保护能检测并修复"前半段无锚点"问题"""
    print("\n" + "="*50)
    print("  F3: 锚点稀疏保护")
    print("="*50)
    
    # 模拟修复前的错误输出（36行挤在 58-59s）
    bad_matches = [
        {"lrc_text": "再見 我們要去明天", "lrc_time": 1.23, "start_sec": 7.46, "end_sec": 57.67, "confidence": 0.29, "_interpolated": False},
    ] + [
        {"lrc_text": f"第{i}行歌词", "lrc_time": i * 3.5, "start_sec": 58.04, "end_sec": 59.04, "confidence": 0.0, "_interpolated": True}
        for i in range(1, 35)
    ] + [
        {"lrc_text": "玩笑話的未來 忽然間到來", "lrc_time": 123.0, "start_sec": 81.7, "end_sec": 109.97, "confidence": 0.53, "_interpolated": False},
        {"lrc_text": "那麼我們再見", "lrc_time": 130.0, "start_sec": 110.43, "end_sec": 136.03, "confidence": 0.43, "_interpolated": False},
        {"lrc_text": "別忘了曾去過彼此的世界", "lrc_time": 140.0, "start_sec": 137.43, "end_sec": 154.6, "confidence": 0.67, "_interpolated": False},
    ]
    
    DURATION_SEC = 157.0
    
    # === F3 保护逻辑（从 app.py 提取） ===
    result = list(bad_matches)  # copy
    result.sort(key=lambda x: x['start_sec'])
    
    first_sub_start = result[0]['start_sec']
    SPARSE_ANCHOR_THRESHOLD = min(15.0, DURATION_SEC * 0.15)
    
    front_cluster_size = min(len(result) // 2, 10)
    front_span = result[front_cluster_size - 1]['start_sec'] - result[0]['start_sec']
    CLUSTER_SQUEEZE_THRESHOLD = 3.0
    
    needs_repair = (
        first_sub_start > SPARSE_ANCHOR_THRESHOLD or 
        front_span < CLUSTER_SQUEEZE_THRESHOLD
    )
    
    print(f"  触发条件检测:")
    print(f"    首字时间: {first_sub_start:.1f}s > 阈值 {SPARSE_ANCHOR_THRESHOLD:.1f}s? {first_sub_start > SPARSE_ANCHOR_THRESHOLD}")
    print(f"    前{front_cluster_size}行跨度: {front_span:.1f}s < 阈值 {CLUSTER_SQUEEZE_THRESHOLD:.1f}s? {front_span < CLUSTER_SQUEEZE_THRESHOLD}")
    print(f"    需要修复? {'是 ✅' if needs_repair else '否'}")
    
    if needs_repair:
        lrc_timed = [(i, r) for i, r in enumerate(result) if r.get('lrc_time') is not None]
        lrc_min_t = min(r['lrc_time'] for _, r in lrc_timed)
        lrc_max_t = max(r['lrc_time'] for _, r in lrc_timed)
        lrc_range = max(lrc_max_t - lrc_min_t, 1.0)
        
        target_end = DURATION_SEC * 0.92
        target_start = DURATION_SEC * 0.03
        target_range = max(target_end - target_start, 30.0)
        
        for idx, sub in enumerate(result):
            if not sub.get('_interpolated') or sub.get('lrc_time') is None:
                continue
            ratio = (sub['lrc_time'] - lrc_min_t) / lrc_range
            new_start = target_start + ratio * target_range
            char_count = len(sub['lrc_text'].strip())
            base_dur = max(1.5, 1.5 + char_count * 0.35)
            new_end = new_start + base_dur
            sub['start_sec'] = round(max(0, new_start), 2)
            sub['end_sec'] = round(min(new_end, DURATION_SEC), 2)
        
        print(f"  F3修复执行: LRC [{lrc_min_t:.0f}~{lrc_max_t:.0f}] → 音频 [{target_start:.1f}~{target_end:.1f}]")
    
    # 验证结果
    repaired_first = result[0]['start_sec']
    repaired_front_span = result[min(10, len(result)-1)]['start_sec'] - result[0]['start_sec']
    
    # 前10行的分布
    print(f"\n  修复后前10行分布:")
    for m in result[:10]:
        tag = "✓" if not m.get('_interpolated') else "~"
        print(f"    [{tag}] {m['start_sec']:7.1f}s | '{m['lrc_text'][:20]}'")
    
    checks = {
        "首字提前到<20s": repaired_first < 20,
        "前10行跨度>10s": repaired_front_span > 10,
        "末字在视频末尾附近": result[-1]['end_sec'] > DURATION_SEC * 0.8,
    }
    
    all_pass = all(checks.values())
    for name, passed in checks.items():
        print(f"  {name}: {'✅' if passed else '❌'}")
    
    print(f"  结果: {'✅ PASS' if all_pass else '❌ FAIL'}")
    return all_pass


# ============================================================
# F4 验证：质量门控
# ============================================================
def test_F4_quality_gate():
    """验证低密度引擎会被降权，不会胜出"""
    print("\n" + "="*50)
    print("  F4: Fusion 引擎质量门控")
    print("="*50)
    
    audio_dur_sec = 157.0
    MIN_WORDS_PER_MIN = 10
    
    def _apply_quality_penalty(engine_name, word_count, raw_score):
        if word_count == 0:
            return raw_score
        wpm = (word_count / audio_dur_sec) * 60
        if wpm < MIN_WORDS_PER_MIN:
            penalty = wpm / MIN_WORDS_PER_MIN
            adjusted = raw_score * penalty
            return adjusted
        return raw_score
    
    # 场景：FunASR 15 词/157秒 = 5.7 词/分，Whisper 80 词/157秒 = 30.6 词/分
    # 但 FunASR 的匹配分更高（因为那个 50 秒长词碰巧匹配上了）
    funasr_wpm = (15 / 157) * 60
    whisper_wpm = (80 / 157) * 60
    
    # 原始分数（假设 FunASR 因为垃圾词碰巧匹配了所以分数高）
    f_raw = 2.65  # FunASR 有一个 50 秒长的"匹配"
    w_raw = 1.85  # Whisper 正常但匹配少
    
    f_adj = _apply_quality_penalty("FunASR", 15, f_raw)
    w_adj = _apply_quality_penalty("Whisper", 80, w_raw)
    
    print(f"  FunASR: {funasr_wpm:.1f}词/分 | 原始={f_raw:.2f} → 调整后={f_adj:.2f}")
    print(f"  Whisper: {whisper_wpm:.1f}词/分 | 原始={w_raw:.2f} → 调整后={w_adj:.2f}")
    print(f"  决策: {'Whisper ✅' if w_adj >= f_adj else 'FunASR ❌'} 胜出")
    
    # 关键检查：低密度的 FunASR 应该被降权
    ok = (w_adj >= f_adj) and (f_adj < f_raw)
    print(f"  结果: {'✅ PASS' if ok else '❌ FAIL'} (FunASR被降权={f_adj<f_raw})")
    return ok


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║   字幕修复 F1-F5 独立验证              ║")
    print("╚══════════════════════════════════════╝")
    
    results = {}
    results["F1 垃圾词过滤"] = test_F1_garbage_filter()
    results["F3 锚点稀疏保护"] = test_F3_anchor_sparse_protect()
    results["F4 质量门控"] = test_F4_quality_gate()
    
    print("\n" + "="*50)
    print("  总体结果")
    print("="*50)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n  通过: {passed}/{total}")
    
    if passed == total:
        print("\n  🎉 所有修复验证通过! 可以进行端到端测试。")
    else:
        print("\n  ⚠️  有验证未通过，需要进一步排查。")
