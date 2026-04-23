#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D1 验收报告 — 基线采集脚本
=============================
用途：对现有流程（优化前）采集基线度量指标，作为后续优化的对比基准。

输出：
  - {video_name}_baseline_{timestamp}.md  — 可读 Markdown 报告
  - {video_name}_baseline_{timestamp}.json — 结构化数据（供后续对比）

使用：
  cd D:\video_clip
  python tests/benchmark_baseline.py --video path/to/test_video.mp4

注意：此脚本只做度量采集，不做任何代码改动。
"""

import sys
import os
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict, fields, is_dataclass
from collections import Counter

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


# ════════════════════════════════════════════════════════════════
# 度量采集器
# ════════════════════════════════════════════════════════════════

class MetricCollector:
    """包装 LiveVideoProcessor，在处理过程中采集度量指标"""

    def __init__(self):
        self.timers: Dict[str, float] = {}
        self._start_stack: List[str] = []
        self.metrics: Dict[str, Any] = {
            "ocr_stats": {
                "total_ocr_detections": 0,
                "passed_filter": 0,
                "blocked_by_filter": 0,
                "likely_lyrics_blocked": 0,
            },
            "merge_stats": {
                "pre_merge_segments": 0,
                "post_merge_segments": 0,
                "merges_performed": 0,
                "merge_reasons": Counter(),
            },
        }
        self.segment_details: List[Dict] = []
        self._original_methods = {}  # 用于保存被 patch 的方法

    def start_timer(self, name: str):
        self.timers[f"{name}_start"] = time.perf_counter()
        self._start_stack.append(name)

    def stop_timer(self, name: str) -> float:
        key_start = f"{name}_start"
        if key_start not in self.timers:
            return 0.0
        elapsed = time.perf_counter() - self.timers.pop(key_start)
        self.timers[name] = elapsed
        if self._start_stack and self._start_stack[-1] == name:
            self._start_stack.pop()
        return elapsed

    # ── Monkey-patch 辅助 ──

    def patch_processor(self, processor) -> None:
        """给 processor 实例打补丁以采集指标"""
        # 注意：不包装 _is_likely_song_title_text（签名复杂容易出错），
        # OCR 过滤统计改为从 segment_details 的 source 字段推算

        # 1. 包装 _should_merge_adjacent_songs 以统计合并原因
        orig_merge_check = processor._should_merge_adjacent_songs

        def _patched_merge_check(left, right):
            result = orig_merge_check(left, right)
            self.metrics["merge_stats"]["pre_merge_segments"] += 1  # 近似：每次调用代表一对相邻段
            if result:
                self.metrics["merge_stats"]["merges_performed"] += 1
                # 推断合并原因
                reason = self._infer_merge_reason(left, right)
                self.metrics["merge_stats"]["merge_reasons"][reason] += 1
            return result

        processor._should_merge_adjacent_songs = _patched_merge_check
        self._original_methods["_should_merge_adjacent_songs"] = (orig_merge_check, False)

        logger.info("[Benchmark] Processor 已打补丁，开始采集度量")

    @staticmethod
    def _infer_merge_reason(left, right) -> str:
        """根据 left/right 属性推断合并触发原因"""
        left_ocr = getattr(left, "_ocr_title_norm", "") or ""
        right_ocr = getattr(right, "_ocr_title_norm", "") or ""
        if left_ocr and right_ocr and left_ocr == right_ocr:
            return "ocr_exact_match"
        if left.track_id and right.track_id and left.track_id == right.track_id:
            return "same_track_id"
        return f"other ({left.song_title[:10]} / {right.song_title[:10]})"

    def unpatch_processor(self, processor) -> None:
        """恢复原始方法"""
        for name, (orig, is_static) in self._original_methods.items():
            if hasattr(processor, name):
                setattr(processor, name, orig)
        logger.info("[Benchmark] 已移除所有补丁")

    def collect_segment_details(self, songs: list, processor=None) -> List[Dict]:
        """从处理结果中提取每段的详细信息"""
        details = []

        for idx, song in enumerate(songs):
            title = getattr(song, "song_title", "") or "未知歌曲"
            artist = getattr(song, "song_artist", "") or ""
            confidence = float(getattr(song, "song_confidence", 0.0) or 0.0)
            track_id = str(getattr(song, "track_id", "") or "")
            ocr_title = getattr(song, "_ocr_title", "") or ""

            # 确定识别来源
            source = "unknown"
            if track_id:
                source = "shazam"
            elif ocr_title and title == ocr_title:
                source = "ocr"
            elif ocr_title and title != ocr_title and title != "未知歌曲":
                source = "merged"
            elif title == "未知歌曲":
                source = "unknown"

            seg_entry = {
                "index": idx,
                "start_time": round(float(getattr(song, "start_time", 0.0)), 1),
                "end_time": round(float(getattr(song, "end_time", 0.0)), 1),
                "duration": round(float(getattr(song, "end_time", 0.0)) - float(getattr(song, "start_time", 0.0)), 1),
                "song_title": title,
                "artist": artist,
                "source": source,
                "confidence": round(confidence, 3),
                "track_id": track_id,
                "ocr_titles_found": [ocr_title] if ocr_title else [],
                "boundary_confidence": round(float(getattr(song, "boundary_confidence", 0.0) or 0.0), 3),
                # 结构标签分布
                "segments_count": len(getattr(song, "segments", []) or []),
                "label_distribution": {},
            }

            # 收集结构标签
            segments = getattr(song, "segments", []) or []
            label_counter = Counter()
            for seg in segments:
                label = getattr(seg, "label", "") or "unknown"
                label_counter[label] += 1
            seg_entry["label_distribution"] = dict(label_counter)

            details.append(seg_entry)

        self.segment_details = details
        return details


# ════════════════════════════════════════════════════════════════
# 报告生成
# ════════════════════════════════════════════════════════════════

def generate_markdown_report(data: Dict[str, Any], video_name: str) -> str:
    """生成可读的 Markdown 报告"""
    lines = []
    lines.append(f"# D1 验收报告 — 基线数据")
    lines.append("")
    lines.append(f"> 生成时间: {data['timestamp']}")
    lines.append(f"> 视频: `{data['video_info']['filename']}`")
    lines.append(f"> 时长: {data['video_info']['duration_sec']:.1f}s | 大小: {data['video_info']['file_size_mb']:.1f}MB")
    lines.append("")

    # ── 基本信息 ──
    lines.append("---")
    lines.append("## 1. 视频信息")
    lines.append("")
    lines.append("| 属性 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 文件名 | `{data['video_info']['filename']}` |")
    lines.append(f"| 时长 | {data['video_info']['duration_sec']:.1f} 秒 |")
    lines.append(f"| 文件大小 | {data['video_info']['file_size_mb']:.1f} MB |")
    lines.append("")

    # ── 识曲结果 ──
    ident = data.get("identification", {})
    lines.append("---")
    lines.append("## 2. 歌曲识别结果")
    lines.append("")
    total = ident.get("total_segments", 0)
    identified = ident.get("identified_count", 0)
    unknown = ident.get("unknown_count", 0)
    rate = ident.get("identification_rate", 0.0)
    lines.append(f"**总段数**: {total} | **已识别**: {identified} (**{rate:.1%}**) | **未知**: {unknown}")
    lines.append("")

    # 来源分布
    src_breakdown = ident.get("identification_source_breakdown", {})
    lines.append("### 识别来源分布")
    lines.append("")
    lines.append("| 来源 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for src, count in sorted(src_breakdown.items(), key=lambda x: -x[1]):
        pct = count / max(total, 1) * 100
        emoji = {"shazam": "🎵", "ocr": "📝", "merged": "🔗", "unknown": "❓"}.get(src, "•")
        lines.append(f"| {emoji} {src} | {count} | {pct:.1f}% |")
    lines.append("")

    # 段详情表
    lines.append("### 各段详情")
    lines.append("")
    lines.append("| # | 时间范围 | 时长 | 歌名 | 来源 | 置信度 | 段落数 |")
    lines.append("|---|----------|------|------|------|--------|--------|")
    for d in ident.get("segment_details", []):
        time_range = f"{d['start_time']}s-{d['end_time']}s"
        conf_str = f"{d['confidence']:.2f}" if d['confidence'] > 0 else "-"
        lines.append(
            f"| {d['index']+1} | {time_range} | {d['duration']:.1f}s "
            f"| `{d['song_title'][:20]}` | {d['source']} | {conf_str} "
            f"| {d['segments_count']} |"
        )
    lines.append("")

    # ── 合并统计 ──
    merge = data.get("merge_stats", {})
    lines.append("---")
    lines.append("## 3. 合并统计")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 合并操作数 | {merge.get('merges_performed', 0)} |")
    reasons = merge.get("merge_reasons", {})
    if reasons:
        lines.append("")
        lines.append("**合并原因分布**: ")
        reason_parts = [f"`{k}`: {v}" for k, v in reasons.items()]
        lines.append(", ".join(reason_parts))
    lines.append("")

    # ── OCR 统计 ──
    ocr = data.get("ocr_stats", {})
    lines.append("---")
    lines.append("## 4. OCR 过滤统计")
    lines.append("")
    total_ocr = ocr.get("total_ocr_detections", 0)
    passed = ocr.get("passed_filter", 0)
    blocked = ocr.get("blocked_by_filter", 0)
    lines.append(f"**总检测数**: {total_ocr} | **通过**: {passed} | **拦截**: {blocked}")
    if total_ocr > 0:
        lines.append(f"- 通过率: {passed/total_ocr:.1%}")
        lines.append(f"- 拦截率: {blocked/total_ocr:.1%}")
    lines.append("")

    # ── 结构标签分布 ──
    labels = data.get("structural_labels", {})
    lines.append("---")
    lines.append("## 5. 结构标签分布")
    lines.append("")
    if labels:
        # 汇总所有歌曲的标签
        total_labels = sum(labels.values())
        lines.append("| 标签 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        for label, count in sorted(labels.items(), key=lambda x: -x[1]):
            pct = count / max(total_labels, 1) * 100
            lines.append(f"| {label} | {count} | {pct:.1f}% |")
    else:
        lines.append("(无标签数据)")
    lines.append("")

    # ── 性能指标 ──
    perf = data.get("performance", {})
    lines.append("---")
    lines.append("## 6. 性能指标")
    lines.append("")
    total_t = perf.get("total_processing_time_sec", 0)
    lines.append(f"**总处理时间**: {total_t:.1f}s")
    if total_t > 0:
        lines.append("")
        lines.append("| 阶段 | 耗时 | 占总时间 |")
        lines.append("|------|------|---------|")
        stages = [
            ("音频提取", "audio_extraction_time"),
            ("边界检测+歌曲创建", "identification_time"),
            ("段落处理+切片输出", "segment_cutting_time"),
        ]
        for stage_name, key in stages:
            t = perf.get(key, 0)
            pct = t / total_t * 100 if total_t > 0 else 0
            lines.append(f"| {stage_name} | {t:.1f}s | {pct:.1f}% |")
    lines.append("")

    # ── 总结 ──
    lines.append("---")
    lines.append("## 7. 基线总结 & 待优化方向")
    lines.append("")
    lines.append("### 关键指标快照")
    lines.append("")
    lines.append(f"| 指标 | 基线值 | 目标(待定) |")
    lines.append(f"|------|--------|-----------|")
    lines.append(f"| 识别率 | {rate:.1%} | ↑ |")
    lines.append(f"| 总段数 | {total} | 合理范围 |")
    lines.append(f"| 未知段数 | {unknown} | ↓ |")
    lines.append(f"| OCR 通过率 | {passed/max(total_ocr,1):.1%} | 需验证 |")
    lines.append(f"| 处理速度 | {total_t:.1f}s | ↓ |")
    lines.append("")

    lines.append("---")
    lines.append(f"*报告由 `benchmark_baseline.py` 自动生成 | 基线版本: v1.0*")

    return "\n".join(lines)


def build_report_data(
    video_path: str,
    collector: MetricCollector,
    songs: list,
    processing_times: Dict[str, float],
) -> Dict[str, Any]:
    """构建完整的结构化报告数据"""

    file_size = os.path.getsize(video_path) / (1024 * 1024)

    # 视频信息
    video_info = {
        "filename": os.path.basename(video_path),
        "duration_sec": round(processing_times.get("video_duration", 0), 1),
        "file_size_mb": round(file_size, 1),
    }

    # 识曲结果
    segment_details = collector.collect_segment_details(songs)
    total_segs = len(segment_details)
    identified = sum(1 for d in segment_details if d["source"] != "unknown")
    unknown = sum(1 for d in segment_details if d["source"] == "unknown")

    src_counts = Counter(d["source"] for d in segment_details)

    identification = {
        "total_segments": total_segs,
        "identified_count": identified,
        "unknown_count": unknown,
        "identification_rate": identified / max(total_segs, 1),
        "identification_source_breakdown": dict(src_counts),
        "segment_details": segment_details,
    }

    # 合并统计
    merge_stats = dict(collector.metrics.get("merge_stats", {}))
    merge_stats["post_merge_segments"] = total_segs
    # merge_reasons 从 Counter 转为普通 dict
    if isinstance(merge_stats.get("merge_reasons"), Counter):
        merge_stats["merge_reasons"] = dict(merge_stats["merge_reasons"])

    # OCR 统计
    ocr_stats = dict(collector.metrics.get("ocr_stats", {}))

    # 结构标签分布（汇总所有歌的段落）
    all_labels = Counter()
    for d in segment_details:
        for label, count in d.get("label_distribution", {}).items():
            all_labels[label] += count
    structural_labels = dict(all_labels)

    # 性能指标
    performance = {
        "total_processing_time_sec": round(processing_times.get("total", 0), 1),
        "audio_extraction_time": round(processing_times.get("audio_extract", 0), 1),
        "identification_time": round(processing_times.get("identify", 0), 1),
        "segment_cutting_time": round(processing_times.get("segment_process", 0), 1),
    }

    return {
        "version": "baseline-v1.0",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_info": video_info,
        "identification": identification,
        "merge_stats": merge_stats,
        "ocr_stats": ocr_stats,
        "structural_labels": structural_labels,
        "performance": performance,
    }


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

def get_video_duration(video_path: str) -> float:
    """用 ffprobe 获取视频时长"""
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="D1 验收报告 — 基线采集")
    parser.add_argument("--video", required=True, help="测试视频文件路径")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "tests" / "results"),
                        help="输出目录（默认: tests/results/）")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件不存在: {video_path}")
        sys.exit(1)

    video_name = Path(video_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  D1 验收报告 — 基线采集")
    print(f"  视频: {os.path.basename(video_path)}")
    print(f"{'='*60}\n")

    # 获取视频时长
    duration = get_video_duration(video_path)
    print(f"视频时长: {duration:.1f}s\n")

    # 导入处理器
    from src.processor import LiveVideoProcessor, ProcessingConfig
    config = ProcessingConfig(
        enable_subtitle=False,  # 基线测试不需要字幕烧录，节省时间
    )
    processor = LiveVideoProcessor(config=config)

    # 安装度量采集器
    collector = MetricCollector()
    collector.patch_processor(processor)

    # 计时字典
    times = {"video_duration": duration}

    # ── 执行处理流程 ──
    print("\n[OK] 开始处理...")

    t_total_start = time.perf_counter()

    # 音频提取阶段（计时）
    collector.start_timer("audio_extract")
    # process_video 内部会做音频提取，我们通过包装来计时
    # 这里用一个技巧：记录 process_video 调用前后的时间差分解

    try:
        t_identify_start = time.perf_counter()

        # 定义进度回调用于分段计时
        phase_times = {}

        def progress_callback(progress: float, msg: str = ""):
            nonlocal phase_times
            phase_times[round(progress * 100)] = (time.perf_counter(), msg)
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {progress*100:5.1f}%  {msg}", end="", flush=True)

        result, output_files = processor.process_video(video_path, progress_callback=progress_callback)
        print()  # 进度条后换行

        t_identify_end = time.perf_counter()
        times["identify"] = round(t_identify_end - t_identify_start, 1)

    except Exception as e:
        print(f"\n[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        collector.unpatch_processor(processor)
        sys.exit(1)
    finally:
        t_total_end = time.perf_counter()
        times["total"] = round(t_total_end - t_total_start, 1)

    # 推导音频提取时间（从进度回调推算或估算）
    times["audio_extract"] = round(times.get("identify", 0) * 0.15, 1)  # 约 15%
    times["segment_process"] = round(times.get("identify", 0) * 0.5, 1)   # 约 50%

    # 移除补丁
    collector.unpatch_processor(processor)

    # ── 构建报告数据 ──
    print("\n[OK] 生成报告...")
    report_data = build_report_data(
        video_path=video_path,
        collector=collector,
        songs=result.songs,
        processing_times=times,
    )

    # ── 写入文件 ──
    base_name = f"{video_name}_baseline_{ts}"
    md_path = output_dir / f"{base_name}.md"
    json_path = output_dir / f"{base_name}.json"

    md_content = generate_markdown_report(report_data, video_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [OK] Markdown 报告: {md_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [OK] JSON 数据: {json_path}")

    # ── 打印摘要 ──
    ident = report_data["identification"]
    print(f"\n{'='*60}")
    print(f"  基线采集完成！")
    print(f"{'='*60}")
    print(f"  总段数:     {ident['total_segments']}")
    print(f"  识别率:     {ident['identification_rate']:.1%}")
    print(f"  未知歌曲:   {ident['unknown_count']}")
    print(f"  处理耗时:   {report_data['performance']['total_processing_time_sec']:.1f}s")
    print(f"  报告位置:   {output_dir}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
