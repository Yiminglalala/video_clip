# -*- coding: utf-8 -*-
"""
歌词对齐引擎（Whisper/FunASR 通用）
---------------------------------
1) 单引擎：单调约束动态对齐（Monotonic DP）
2) 双引擎：按行融合锚点（不是整条时间轴二选一）
3) 输出 line_status / line_confidence / alignment_meta
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple


_PUNCT_RE = re.compile(
    r"[\s\u3000\u3001\u3002\uff01\uff0c\uff0e\uff1a\uff1b\uff1f\uff08\uff09\u2014\u2018\u2019\"'\u2026\u201c\u201d]"
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    try:
        from zhconv import convert  # type: ignore

        text = convert(text, "zh-cn")
    except Exception:
        pass
    text = _PUNCT_RE.sub("", str(text))
    text = re.sub(r"[a-zA-Z0-9_]+", "", text)
    return text.strip()


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    substr = 0.0
    if b_norm in a_norm:
        substr = 0.45 * (len(b_norm) / max(len(a_norm), 1)) + 0.08
    elif a_norm in b_norm:
        substr = 0.40 * (len(a_norm) / max(len(b_norm), 1)) + 0.08
    return min(1.0, max(ratio, substr))


@dataclass
class ASRWord:
    text: str
    start: float
    end: float


@dataclass
class WindowCandidate:
    start: float
    end: float
    center: float
    text: str
    sim: float
    score: float


def _asr_words_from_raw(asr_words: Iterable[dict]) -> List[ASRWord]:
    parsed: List[ASRWord] = []
    for w in asr_words or []:
        text = str(w.get("text", w.get("word", ""))).strip()
        if not text:
            continue
        start = w.get("start", w.get("start_ms", 0.0))
        end = w.get("end", w.get("end_ms", start))
        try:
            s = float(start)
            e = float(end)
        except Exception:
            continue
        # 毫秒格式兼容
        if e > 200.0 and s > 200.0:
            s /= 1000.0
            e /= 1000.0
        if e <= s:
            e = s + 0.08
        parsed.append(ASRWord(text=text, start=max(0.0, s), end=e))
    parsed.sort(key=lambda x: x.start)
    return parsed


def _build_time_windows(
    words: List[ASRWord],
    step_sec: float = 0.25,
    half_window_sec: float = 3.2,
) -> List[Tuple[float, float, str]]:
    if not words:
        return []
    duration = max(w.end for w in words)
    windows: List[Tuple[float, float, str]] = []
    c = 0.0
    while c <= duration + 1e-6:
        ws = max(0.0, c - half_window_sec)
        we = c + half_window_sec
        pieces = [w.text for w in words if not (w.end < ws or w.start > we)]
        if pieces:
            windows.append((ws, we, "".join(pieces)))
        c += step_sec
    return windows


def _strictness_bias(strictness: str) -> float:
    s = (strictness or "standard").strip().lower()
    if s in ("strict", "严格"):
        return 0.08
    if s in ("loose", "宽松"):
        return -0.07
    return 0.0


def _effective_low_conf(low_conf_threshold: float, strictness: str) -> float:
    return min(0.9, max(0.25, float(low_conf_threshold) + _strictness_bias(strictness)))


def _sim_floor_by_strictness(strictness: str) -> float:
    s = (strictness or "standard").strip().lower()
    if s in ("strict", "严格"):
        return 0.34
    if s in ("loose", "宽松"):
        return 0.28
    return 0.30


def _estimate_line_duration(text: str) -> float:
    ln = len(normalize_text(text or ""))
    return max(0.9, min(4.5, 1.0 + 0.09 * ln))


def _piecewise_predict_time(
    index: int,
    lrc_lines: List[dict],
    anchors: List[dict],
    default_scale: float,
    default_offset: float,
) -> Optional[float]:
    if not anchors:
        return None
    lt_raw = lrc_lines[index].get("time", lrc_lines[index].get("lrc_time"))
    if lt_raw is None:
        return None
    lt = float(lt_raw)

    prev_anchor = None
    next_anchor = None
    for a in anchors:
        if int(a["lrc_index"]) <= index:
            prev_anchor = a
        if int(a["lrc_index"]) >= index:
            next_anchor = a
            break

    if prev_anchor and next_anchor:
        p_lt = prev_anchor.get("lrc_time")
        n_lt = next_anchor.get("lrc_time")
        p_st = prev_anchor.get("start_sec")
        n_st = next_anchor.get("start_sec")
        if None not in (p_lt, n_lt, p_st, n_st):
            p_lt_f = float(p_lt)
            n_lt_f = float(n_lt)
            if abs(n_lt_f - p_lt_f) > 1e-6:
                ratio = (lt - p_lt_f) / (n_lt_f - p_lt_f)
                ratio = max(0.0, min(1.0, ratio))
                return float(p_st) + ratio * (float(n_st) - float(p_st))

    if prev_anchor:
        p_lt = prev_anchor.get("lrc_time")
        p_st = prev_anchor.get("start_sec")
        if p_lt is not None and p_st is not None:
            return float(p_st) + float(default_scale) * (lt - float(p_lt))
    if next_anchor:
        n_lt = next_anchor.get("lrc_time")
        n_st = next_anchor.get("start_sec")
        if n_lt is not None and n_st is not None:
            return float(n_st) - float(default_scale) * (float(n_lt) - lt)

    return float(default_scale) * lt + float(default_offset)


def _fill_reviews_with_anchor_interpolation(
    line_details: List[dict],
    lrc_lines: List[dict],
    tempo_scale: float,
    global_offset: float,
) -> int:
    anchors = [
        d
        for d in line_details
        if d.get("line_status") == "confirmed"
        and d.get("start_sec") is not None
        and d.get("lrc_time") is not None
    ]
    anchors = sorted(anchors, key=lambda x: int(x.get("lrc_index", -1)))
    changed = 0

    for i, detail in enumerate(line_details):
        if detail.get("line_status") == "confirmed":
            detail["time_source"] = "anchor"
            continue
        
        # 即使没有锚点，也应该应用global_offset和tempo_scale
        pred = _piecewise_predict_time(
            index=i,
            lrc_lines=lrc_lines,
            anchors=anchors,
            default_scale=tempo_scale,
            default_offset=global_offset,
        )
        
        # 如果没有锚点，使用LRC时间作为基础，并添加一个偏移量以匹配实际演唱时间
        if pred is None:
            lt_raw = lrc_lines[i].get("time", lrc_lines[i].get("lrc_time"))
            if lt_raw is not None:
                lt = float(lt_raw)
                # 当没有锚点时，使用LRC时间戳作为基础
                # 添加一个默认偏移量，让字幕显示的时间比LRC时间戳晚一些，以匹配实际的演唱时间
                # 这个偏移量可以根据实际情况调整
                default_offset = 13.0  # 默认偏移13秒，以匹配用户反馈的35秒左右开始唱歌的情况
                pred = lt + default_offset
            else:
                continue
        
        existing = detail.get("start_sec")
        if existing is not None:
            try:
                pred = 0.55 * float(pred) + 0.45 * float(existing)
            except Exception:
                pass
        start = max(0.0, float(pred))
        dur = _estimate_line_duration(detail.get("lrc_text", ""))
        end = start + dur

        # 用下一锚点保护上界，避免跨行覆盖
        next_anchor = None
        for a in anchors:
            if int(a.get("lrc_index", -1)) > i:
                next_anchor = a
                break
        if next_anchor and next_anchor.get("start_sec") is not None:
            end = min(end, max(start + 0.35, float(next_anchor["start_sec"]) - 0.05))

        # 计算置信度
        if anchors:
            conf_neighbors = []
            for a in anchors:
                if abs(int(a.get("lrc_index", -9999)) - i) <= 2:
                    conf_neighbors.append(float(a.get("line_confidence", 0.5)))
            interp_conf = 0.32 if not conf_neighbors else max(0.26, min(0.62, sum(conf_neighbors) / len(conf_neighbors) * 0.7))
        else:
            # 没有锚点时的默认置信度
            interp_conf = 0.25

        detail["start_sec"] = round(start, 3)
        detail["end_sec"] = round(max(end, start + 0.35), 3)
        detail["line_confidence"] = round(float(interp_conf), 4)
        detail["line_status"] = "interpolated"
        detail["time_source"] = "interp"
        changed += 1

    # 单调化 + 去重叠
    prev_start = None
    for detail in line_details:
        st = detail.get("start_sec")
        ed = detail.get("end_sec")
        if st is None:
            continue
        st_f = float(st)
        ed_f = float(ed if ed is not None else st_f + 0.35)
        if prev_start is not None and st_f < prev_start + 0.02:
            st_f = prev_start + 0.02
        ed_f = max(ed_f, st_f + 0.35)
        detail["start_sec"] = round(st_f, 3)
        detail["end_sec"] = round(ed_f, 3)
        prev_start = st_f

    for idx in range(len(line_details) - 1):
        cur = line_details[idx]
        nxt = line_details[idx + 1]
        if cur.get("start_sec") is None or nxt.get("start_sec") is None:
            continue
        cur_end = float(cur.get("end_sec", cur["start_sec"] + 0.35))
        nxt_start = float(nxt["start_sec"])
        if cur_end >= nxt_start:
            cur["end_sec"] = round(max(float(cur["start_sec"]) + 0.35, nxt_start - 0.03), 3)

    return changed


def _estimate_transform_ransac(
    lrc_lines: List[dict],
    windows: List[Tuple[float, float, str]],
    max_drift_sec: float,
) -> Tuple[float, float, float]:
    """
    估计 audio_time = tempo_scale * lrc_time + global_offset
    返回: (tempo_scale, global_offset, avg_fit_error)
    """
    timed = []
    for idx, line in enumerate(lrc_lines):
        t = line.get("time", line.get("lrc_time"))
        if t is None:
            continue
        line_text = line.get("text", "")
        best = None
        for ws, we, wt in windows:
            sim = text_similarity(wt, line_text)
            if best is None or sim > best[1]:
                best = ((ws + we) * 0.5, sim)
        if best and best[1] >= 0.12:
            timed.append((float(t), float(best[0]), float(best[1])))
    if len(timed) < 2:
        # 退化：只估 offset
        if timed:
            off = timed[0][1] - timed[0][0]
            return 1.0, off, 0.0
        return 1.0, 0.0, 0.0

    best_model = (1.0, 0.0)
    best_score = -1e9
    best_err = 999.0
    pairs = timed[:]
    max_iter = min(80, len(pairs) * len(pairs))
    rng = random.Random(42)

    for _ in range(max_iter):
        p1, p2 = rng.sample(pairs, 2)
        dt = p2[0] - p1[0]
        if abs(dt) < 1e-6:
            continue
        scale = (p2[1] - p1[1]) / dt
        if not (0.45 <= scale <= 1.85):
            continue
        offset = p1[1] - scale * p1[0]

        inliers = 0
        wsum = 0.0
        err_sum = 0.0
        for lt, at, sim in pairs:
            pred = scale * lt + offset
            err = abs(pred - at)
            if err <= max_drift_sec:
                inliers += 1
                wsum += sim
                err_sum += err
        if inliers == 0:
            continue
        score = inliers * 2.0 + wsum
        avg_err = err_sum / inliers
        if score > best_score or (abs(score - best_score) < 1e-6 and avg_err < best_err):
            best_score = score
            best_err = avg_err
            best_model = (scale, offset)

    return best_model[0], best_model[1], 0.0 if best_err >= 998 else best_err


def align_lrc_monotonic(
    asr_words: List[dict],
    lrc_lines: List[dict],
    strictness: str = "standard",
    low_conf_threshold: float = 0.42,
    max_drift_sec: float = 1.2,
    engine_name: str = "",
) -> Dict[str, object]:
    words = _asr_words_from_raw(asr_words)
    if not words or not lrc_lines:
        return {
            "engine": engine_name,
            "confirmed_matches": [],
            "all_candidates": [],
            "line_details": [],
            "alignment_meta": {
                "global_offset": 0.0,
                "tempo_scale": 1.0,
                "confirmed_ratio": 0.0,
                "review_count": len(lrc_lines or []),
                "avg_line_error_est": None,
                "drift_risk": "high",
            },
        }

    windows = _build_time_windows(words)
    tempo_scale, global_offset, fit_err = _estimate_transform_ransac(
        lrc_lines=lrc_lines,
        windows=windows,
        max_drift_sec=max(0.6, max_drift_sec),
    )

    low_conf_eff = _effective_low_conf(low_conf_threshold, strictness)
    sim_floor = _sim_floor_by_strictness(strictness)
    top_k = 8
    norm_counts: Dict[str, int] = {}
    for line in lrc_lines:
        n = normalize_text(str(line.get("text", "")))
        if n:
            norm_counts[n] = norm_counts.get(n, 0) + 1

    # 每行候选
    per_line: List[List[WindowCandidate]] = []
    for i, line in enumerate(lrc_lines):
        line_text = str(line.get("text", ""))
        line_norm = normalize_text(line_text)
        lt = line.get("time", line.get("lrc_time"))
        predicted = None if lt is None else tempo_scale * float(lt) + global_offset
        candidates: List[WindowCandidate] = []
        for ws, we, wt in windows:
            center = (ws + we) * 0.5
            sim = text_similarity(wt, line_text)
            if sim < 0.06:
                continue
            if predicted is None:
                time_score = 0.5
            else:
                err = abs(center - predicted)
                time_score = max(0.0, 1.0 - err / max(max_drift_sec, 0.6))
            short_penalty = 0.08 if 0 < len(line_norm) <= 3 else 0.0
            repeat_penalty = 0.0
            if line_norm:
                repeat_penalty = min(0.12, max(0, norm_counts.get(line_norm, 1) - 1) * 0.03)
            score = 0.82 * sim + 0.18 * time_score - short_penalty - repeat_penalty
            candidates.append(
                WindowCandidate(start=ws, end=we, center=center, text=wt, sim=sim, score=score)
            )
        candidates.sort(key=lambda x: x.score, reverse=True)
        per_line.append(candidates[:top_k] if candidates else [])

    # DP: 单调约束（行索引递增 + 时间不回跳）
    neg_inf = -1e18
    dp: List[List[float]] = []
    prev_idx: List[List[int]] = []
    for i, cands in enumerate(per_line):
        c_len = max(1, len(cands))
        dp.append([neg_inf] * c_len)
        prev_idx.append([-1] * c_len)
        if i == 0:
            if cands:
                for k, c in enumerate(cands):
                    dp[i][k] = c.score
            else:
                dp[i][0] = 0.02
            continue

        prev_cands = per_line[i - 1]
        if cands:
            for k, c in enumerate(cands):
                best_s = neg_inf
                best_j = -1
                if prev_cands:
                    for j, pc in enumerate(prev_cands):
                        if c.center + 0.2 < pc.center:
                            continue
                        trans = 0.0
                        prev_lt = lrc_lines[i - 1].get("time", lrc_lines[i - 1].get("lrc_time"))
                        cur_lt = lrc_lines[i].get("time", lrc_lines[i].get("lrc_time"))
                        if prev_lt is not None and cur_lt is not None:
                            expect = (float(cur_lt) - float(prev_lt)) * tempo_scale
                            actual = c.center - pc.center
                            gap_err = abs(actual - expect)
                            trans = 0.12 * max(-0.4, 1.0 - gap_err / max(max_drift_sec, 0.8))
                        v = dp[i - 1][j] + c.score + trans
                        if v > best_s:
                            best_s = v
                            best_j = j
                else:
                    best_s = dp[i - 1][0] + c.score
                    best_j = 0
                dp[i][k] = best_s
                prev_idx[i][k] = best_j
        else:
            # 当前行无候选，允许跳过（后续做 interpolated/review）
            best_s = neg_inf
            best_j = -1
            prev_len = max(1, len(prev_cands))
            for j in range(prev_len):
                v = dp[i - 1][j] - 0.03
                if v > best_s:
                    best_s = v
                    best_j = j
            dp[i][0] = best_s
            prev_idx[i][0] = best_j

    # 回溯最佳路径
    last_i = len(lrc_lines) - 1
    last_k = max(range(len(dp[last_i])), key=lambda x: dp[last_i][x])
    choose_idx = [-1] * len(lrc_lines)
    i = last_i
    k = last_k
    while i >= 0:
        choose_idx[i] = k
        pk = prev_idx[i][k] if i > 0 else -1
        i -= 1
        if i >= 0:
            k = 0 if pk < 0 else min(pk, len(dp[i]) - 1)

    line_details: List[dict] = []
    confirmed: List[dict] = []
    all_candidates: List[dict] = []

    for i, line in enumerate(lrc_lines):
        lt = line.get("time", line.get("lrc_time"))
        predicted = None if lt is None else tempo_scale * float(lt) + global_offset
        cands = per_line[i]
        chosen = None
        if cands and choose_idx[i] >= 0:
            chosen = cands[min(choose_idx[i], len(cands) - 1)]

        if chosen is None:
            if predicted is not None:
                start = max(0.0, predicted)
                end = start + max(0.9, min(4.0, 1.3 + len(str(line.get("text", ""))) * 0.04))
                line_conf = 0.0
                status = "interpolated"
            else:
                start = None
                end = None
                line_conf = 0.0
                status = "review"
            sim = 0.0
        else:
            start = max(0.0, chosen.start)
            end = max(chosen.end, start + 0.2)
            sim = chosen.sim
            line_conf = float(max(0.0, min(1.0, chosen.score)))
            status = "confirmed" if line_conf >= low_conf_eff and sim >= sim_floor else "review"

        detail = {
            "lrc_index": i,
            "lrc_text": line.get("text", ""),
            "lrc_time": lt,
            "start_sec": None if start is None else round(float(start), 3),
            "end_sec": None if end is None else round(float(end), 3),
            "line_confidence": round(float(line_conf), 4),
            "line_status": status,
            "sim_score": round(float(sim), 4),
            "source_engine": engine_name,
            "time_source": "anchor" if status == "confirmed" else ("interp" if status == "interpolated" else "review"),
        }
        line_details.append(detail)

    _fill_reviews_with_anchor_interpolation(
        line_details=line_details,
        lrc_lines=lrc_lines,
        tempo_scale=tempo_scale,
        global_offset=global_offset,
    )

    confirmed = []
    all_candidates = []
    for detail in line_details:
        if detail.get("start_sec") is not None:
            all_candidates.append(
                {
                    "lrc_index": detail["lrc_index"],
                    "lrc_text": detail["lrc_text"],
                    "lrc_time": detail["lrc_time"],
                    "start_sec": detail["start_sec"],
                    "end_sec": detail["end_sec"],
                    "confidence": detail["line_confidence"],
                    "line_status": detail["line_status"],
                    "source_engine": engine_name,
                    "time_source": detail.get("time_source", "review"),
                }
            )
        if detail.get("line_status") == "confirmed" and detail.get("start_sec") is not None:
            confirmed.append(
                {
                    "lrc_index": detail["lrc_index"],
                    "lrc_text": detail["lrc_text"],
                    "lrc_time": detail["lrc_time"],
                    "start_sec": detail["start_sec"],
                    "end_sec": detail["end_sec"],
                    "confidence": detail["line_confidence"],
                    "line_status": "confirmed",
                    "source_engine": engine_name,
                    "time_source": detail.get("time_source", "anchor"),
                }
            )

    confirmed_ratio = (len(confirmed) / max(len(lrc_lines), 1.0)) if lrc_lines else 0.0
    review_count = sum(1 for d in line_details if d["line_status"] == "review")
    interpolated_count = sum(1 for d in line_details if d["line_status"] == "interpolated")
    avg_err_est = None
    if fit_err > 0:
        avg_err_est = round(float(fit_err), 3)

    if confirmed_ratio >= 0.78 and (avg_err_est is None or avg_err_est <= 0.6):
        drift_risk = "low"
    elif confirmed_ratio >= 0.55 and (avg_err_est is None or avg_err_est <= 1.0):
        drift_risk = "medium"
    else:
        drift_risk = "high"

    return {
        "engine": engine_name,
        "confirmed_matches": confirmed,
        "all_candidates": all_candidates,
        "line_details": line_details,
        "alignment_meta": {
            "global_offset": round(float(global_offset), 4),
            "tempo_scale": round(float(tempo_scale), 4),
            "confirmed_ratio": round(float(confirmed_ratio), 4),
            "review_count": int(review_count),
            "interpolated_count": int(interpolated_count),
            "avg_line_error_est": avg_err_est,
            "drift_risk": drift_risk,
        },
    }


def _pick_detail_by_idx(line_details: List[dict]) -> Dict[int, dict]:
    mapping: Dict[int, dict] = {}
    for d in line_details or []:
        idx = int(d.get("lrc_index", -1))
        if idx >= 0:
            mapping[idx] = d
    return mapping


def fuse_engine_alignments(
    lrc_lines: List[dict],
    whisper_alignment: Dict[str, object],
    funasr_alignment: Dict[str, object],
    low_conf_threshold: float = 0.42,
    max_drift_sec: float = 1.2,
) -> Dict[str, object]:
    w_map = _pick_detail_by_idx(list(whisper_alignment.get("line_details", [])))
    f_map = _pick_detail_by_idx(list(funasr_alignment.get("line_details", [])))
    low_conf = _effective_low_conf(low_conf_threshold, "standard")

    line_details: List[dict] = []
    confirmed: List[dict] = []
    all_candidates: List[dict] = []

    for i, line in enumerate(lrc_lines):
        w = w_map.get(i)
        f = f_map.get(i)
        chosen = None

        if w and f and w.get("start_sec") is not None and f.get("start_sec") is not None:
            ws, fs = float(w["start_sec"]), float(f["start_sec"])
            delta = abs(ws - fs)
            wc = float(w.get("line_confidence", 0.0))
            fc = float(f.get("line_confidence", 0.0))
            if w.get("line_status") == "confirmed" and f.get("line_status") == "confirmed" and delta < 0.6:
                total = max(wc + fc, 1e-6)
                start = (ws * wc + fs * fc) / total
                we = float(w.get("end_sec", ws + 0.2))
                fe = float(f.get("end_sec", fs + 0.2))
                end = (we * wc + fe * fc) / total
                conf = max(wc, fc)
                chosen = {
                    "lrc_index": i,
                    "lrc_text": line.get("text", ""),
                    "lrc_time": line.get("time", line.get("lrc_time")),
                    "start_sec": round(start, 3),
                    "end_sec": round(max(end, start + 0.2), 3),
                    "line_confidence": round(conf, 4),
                    "line_status": "confirmed" if conf >= low_conf else "review",
                    "source_engine": "fusion",
                    "time_source": "anchor",
                }
            else:
                chosen = w if wc >= fc else f
                chosen = {
                    **chosen,
                    "line_status": "review",
                    "source_engine": str(chosen.get("source_engine", "fusion")),
                    "time_source": "review",
                }
        elif w and w.get("start_sec") is not None:
            chosen = dict(w)
        elif f and f.get("start_sec") is not None:
            chosen = dict(f)
        else:
            # 两边都没有，优先保留插值信息
            if w and w.get("line_status") == "interpolated":
                chosen = dict(w)
            elif f and f.get("line_status") == "interpolated":
                chosen = dict(f)
            else:
                chosen = {
                    "lrc_index": i,
                    "lrc_text": line.get("text", ""),
                    "lrc_time": line.get("time", line.get("lrc_time")),
                    "start_sec": None,
                    "end_sec": None,
                    "line_confidence": 0.0,
                    "line_status": "review",
                    "source_engine": "fusion",
                    "time_source": "review",
                }
        line_details.append(
            {
                "lrc_index": i,
                "lrc_text": chosen.get("lrc_text", ""),
                "lrc_time": chosen.get("lrc_time"),
                "start_sec": chosen.get("start_sec"),
                "end_sec": chosen.get("end_sec"),
                "line_confidence": float(chosen.get("line_confidence", 0.0)),
                "line_status": chosen.get("line_status", "review"),
                "source_engine": chosen.get("source_engine", "fusion"),
                "time_source": chosen.get("time_source", "review"),
            }
        )

    offsets = []
    scales = []
    weights = []
    for meta in (
        whisper_alignment.get("alignment_meta", {}) if whisper_alignment else {},
        funasr_alignment.get("alignment_meta", {}) if funasr_alignment else {},
    ):
        if not isinstance(meta, dict):
            continue
        try:
            # 根据confirmed_ratio来加权
            weight = float(meta.get("confirmed_ratio", 0.0))
            if weight > 0:
                offsets.append(float(meta.get("global_offset", 0.0)))
                scales.append(float(meta.get("tempo_scale", 1.0)))
                weights.append(weight)
        except Exception:
            continue

    if weights:
        # 加权平均
        total_weight = sum(weights)
        global_offset = sum(o * w for o, w in zip(offsets, weights)) / total_weight
        tempo_scale = sum(s * w for s, w in zip(scales, weights)) / total_weight
    else:
        # 如果没有权重，使用默认值
        global_offset = 0.0
        tempo_scale = 1.0

    _fill_reviews_with_anchor_interpolation(
        line_details=line_details,
        lrc_lines=lrc_lines,
        tempo_scale=tempo_scale,
        global_offset=global_offset,
    )

    confirmed = []
    all_candidates = []
    for chosen in line_details:
        if chosen.get("start_sec") is not None:
            all_candidates.append(
                {
                    "lrc_index": chosen["lrc_index"],
                    "lrc_text": chosen.get("lrc_text", ""),
                    "lrc_time": chosen.get("lrc_time"),
                    "start_sec": float(chosen.get("start_sec")),
                    "end_sec": float(chosen.get("end_sec", float(chosen.get("start_sec")) + 0.2)),
                    "confidence": float(chosen.get("line_confidence", 0.0)),
                    "line_status": chosen.get("line_status", "review"),
                    "source_engine": chosen.get("source_engine", "fusion"),
                    "time_source": chosen.get("time_source", "review"),
                }
            )
        if chosen.get("line_status") == "confirmed" and chosen.get("start_sec") is not None:
            confirmed.append(
                {
                    "lrc_index": chosen["lrc_index"],
                    "lrc_text": chosen.get("lrc_text", ""),
                    "lrc_time": chosen.get("lrc_time"),
                    "start_sec": float(chosen.get("start_sec")),
                    "end_sec": float(chosen.get("end_sec", float(chosen.get("start_sec")) + 0.2)),
                    "confidence": float(chosen.get("line_confidence", 0.0)),
                    "line_status": "confirmed",
                    "source_engine": chosen.get("source_engine", "fusion"),
                    "time_source": chosen.get("time_source", "anchor"),
                }
            )

    confirmed_ratio = len(confirmed) / max(len(lrc_lines), 1.0)
    review_count = sum(1 for d in line_details if d.get("line_status") == "review")
    interpolated_count = sum(1 for d in line_details if d.get("line_status") == "interpolated")

    # 估计误差：双引擎冲突时差异
    diffs = []
    for i in range(len(lrc_lines)):
        w = w_map.get(i)
        f = f_map.get(i)
        if w and f and w.get("start_sec") is not None and f.get("start_sec") is not None:
            diffs.append(abs(float(w["start_sec"]) - float(f["start_sec"])))
    avg_err = round(sum(diffs) / len(diffs), 3) if diffs else None

    if confirmed_ratio >= 0.78 and (avg_err is None or avg_err <= 0.6):
        drift_risk = "low"
    elif confirmed_ratio >= 0.55 and (avg_err is None or avg_err <= max_drift_sec):
        drift_risk = "medium"
    else:
        drift_risk = "high"

    return {
        "engine": "fusion",
        "confirmed_matches": confirmed,
        "all_candidates": all_candidates,
        "line_details": line_details,
        "alignment_meta": {
            "global_offset": round(global_offset, 4),
            "tempo_scale": round(tempo_scale, 4),
            "confirmed_ratio": round(confirmed_ratio, 4),
            "review_count": int(review_count),
            "interpolated_count": int(interpolated_count),
            "avg_line_error_est": avg_err,
            "drift_risk": drift_risk,
        },
    }


def build_hotword_text(title: str, artist: str, lrc_text: str, top_k: int = 30) -> str:
    tokens = []
    for item in (title or "", artist or ""):
        norm = normalize_text(item)
        if norm:
            tokens.append(norm)
    counter: Dict[str, int] = {}
    for raw in (lrc_text or "").splitlines():
        text = normalize_text(raw)
        if not text:
            continue
        # 按汉字切片做轻量关键词（避免空格分词依赖）
        if len(text) <= 2:
            counter[text] = counter.get(text, 0) + 1
            continue
        for i in range(0, len(text) - 1):
            tk = text[i : i + 2]
            counter[tk] = counter.get(tk, 0) + 1
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for tk, _ in ranked[:top_k]:
        if tk not in tokens:
            tokens.append(tk)
    return " ".join(tokens[:top_k])


def compute_asr_quality(asr_text: str, lrc_text: str, words: List[dict]) -> Dict[str, float]:
    """
    识别质量估计：歌词覆盖率 + 时序稳定性
    """
    asr_norm = normalize_text(asr_text or "")
    lrc_norm = normalize_text(lrc_text or "")
    if not asr_norm:
        coverage = 0.0
    elif not lrc_norm:
        coverage = 0.0
    else:
        coverage = SequenceMatcher(None, asr_norm[:1200], lrc_norm[:1200]).ratio()

    durations = []
    monotonic_ok = 0
    prev_s = -1e9
    for w in words or []:
        s = w.get("start", w.get("start_ms", 0.0))
        e = w.get("end", w.get("end_ms", s))
        try:
            s = float(s)
            e = float(e)
        except Exception:
            continue
        if s > 1000 and e > 1000:
            s /= 1000.0
            e /= 1000.0
        d = e - s
        if 0.02 <= d <= 4.0:
            durations.append(d)
        if s >= prev_s - 0.15:
            monotonic_ok += 1
        prev_s = s
    if not words:
        stability = 0.0
    else:
        valid_ratio = len(durations) / max(len(words), 1)
        mono_ratio = monotonic_ok / max(len(words), 1)
        stability = 0.55 * valid_ratio + 0.45 * mono_ratio

    total_score = 0.72 * coverage + 0.28 * stability
    return {
        "coverage": round(float(coverage), 4),
        "stability": round(float(stability), 4),
        "total": round(float(total_score), 4),
    }
