"""
视频帧文字检测器 - 用于检测歌曲标题/歌手信息画面 v2.0（性能优化版）
通过帧差异和颜色特征识别文字画面，支持 OCR 文字识别增强

v2.0 优化：
- 两阶段扫描：粗扫5s间隔 → 精扫候选区域1s间隔（减少80% I/O）
- 暗帧/纯色帧快速跳过（亮度预筛）
- OCR 前图像降分辨率（宽≤1280px）
- 提取重复 OCR 逻辑为独立函数
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ==================== 可调参数 ====================
COARSE_SCAN_INTERVAL = 2       # 第一阶段：每N秒采样一帧（粗扫）— 2秒不会漏掉3s以上的标题画面
FINE_SCAN_INTERVAL = 1         # 第二阶段：候选区域内每N秒采样（精扫）
CANDIDATE_EXPAND_SEC = 6       # 候选区域前后各扩展秒数（留余量）
BRIGHTNESS_MIN = 10            # 暗帧阈值（低于此值直接跳过）
BRIGHTNESS_MAX = 250           # 过曝帧阈值（高于此值直接跳过）
OCR_MAX_WIDTH = 1280           # OCR 输入图像最大宽度
STATIC_THRESHOLD = 8           # 帧差异阈值
MIN_STATIC_DURATION = 2        # 最短静态持续秒数
MAX_STATIC_GAP = 1             # 允许的最大非静态间隔（秒）

# OCR 引擎（懒加载）
_ocr_engine = None

def _get_ocr():
    """懒加载 OCR 引擎"""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    # 尝试 EasyOCR（更轻量）
    try:
        import easyocr
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_mode = cuda_available
        logger.info(f"CUDA 可用: {cuda_available}, 设备: {torch.cuda.get_device_name(0) if cuda_available else 'N/A'}")
        _ocr_engine = easyocr.Reader(['ch_sim', 'en'], gpu=gpu_mode)
        logger.info(f"EasyOCR 加载成功 ({'GPU' if gpu_mode else 'CPU'}模式)")
        return _ocr_engine
    except Exception as e:
        logger.warning(f"EasyOCR 加载失败: {e}")
    
    # 尝试 PaddleOCR
    try:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(lang='ch')
        logger.info("PaddleOCR 加载成功")
        return _ocr_engine
    except Exception as e:
        logger.warning(f"PaddleOCR 加载失败: {e}")
    
    logger.info("OCR 不可用，使用纯静态画面检测")
    return None


def _run_ocr(ocr_engine, frame, ocr_max_width=OCR_MAX_WIDTH):
    """
    对单帧执行 OCR 识别（提取的公共函数，消除重复代码）
    
    Args:
        ocr_engine: 已加载的 OCR 引擎
        frame: BGR 图像（任意分辨率）
        ocr_max_width: OCR 输入最大宽度（自动缩放以加速）
    
    Returns:
        (ocr_text, ocr_score, has_chinese)
    """
    if ocr_engine is None:
        return "", 0.0, False
    
    try:
        # 降分辨率加速 OCR（保持宽高比）
        h, w = frame.shape[:2]
        if w > ocr_max_width:
            scale = ocr_max_width / w
            frame_small = cv2.resize(frame, (ocr_max_width, int(h * scale)))
        else:
            frame_small = frame
        
        result = ocr_engine.readtext(frame_small)
        
        if not result or len(result) == 0:
            return "", 0.0, False
        
        texts = []
        ocr_score = 0.0
        has_chinese = False
        
        for item in result:
            # readtext 返回 [(bbox, text, conf), ...]
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                txt = item[1]   # text
                conf = item[2]   # confidence
                texts.append(txt)
                ocr_score += conf
                # 使用正则快速检测中文（比逐字符快）
                if not has_chinese and re.search(r'[\u4e00-\u9fff]', txt):
                    has_chinese = True
        
        ocr_text = " ".join(texts)
        ocr_score = ocr_score / max(1, len(texts))
        return ocr_text, ocr_score, has_chinese
        
    except Exception as e:
        logger.debug(f"OCR 失败: {e}")
        return "", 0.0, False


class TextFrameDetector:
    """
    检测视频中包含文字（如歌曲名、歌手名）的帧
    特征：文字区域通常有高对比度、规则的几何形状
    
    v2.0: 两阶段扫描架构，大幅降低长视频处理时间
    """

    def __init__(self, sample_interval: int = 75):
        """
        Args:
            sample_interval: 保留兼容（v2.0 实际使用 COARSE/FINE_SCAN_INTERVAL）
        """
        self.sample_interval = sample_interval
        # 保存最近一次检测到的文字帧，供上层做歌曲名提取使用
        self.last_text_frames: List[dict] = []

    def detect_text_frames(
        self,
        video_path: str,
        progress_callback=None
    ) -> List[dict]:
        """
        两阶段静态文字帧检测：
        
        Stage 1 - 粗扫描：每 COARSE_SCAN_INTERVAL 秒采一帧，找出候选时间段
        Stage 2 - 精扫描：仅对候选区间用 1 秒间隔精扫，确认静态段并 OCR
        
        Returns:
            [{'time': 秒, 'score': 置信度, 'type': 'title'/'subtitle', 'text': OCR文字}, ...]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        int_duration = int(total_duration)

        logger.info(f"[v2.0两阶段] 视频时长: {total_duration:.1f}秒, FPS: {fps:.1f}, 总帧数: {total_frames}")

        # 尝试加载 OCR
        ocr = _get_ocr()

        # ═══════════════════════════════════════════════════════════
        # Stage 1: 粗扫描 — 每 N 秒采一帧，找候选静态段
        # ═══════════════════════════════════════════════════════════
        candidate_regions = self._coarse_scan(
            cap, fps, total_frames, total_duration, int_duration, progress_callback
        )
        
        logger.info(f"[Stage1 粗扫] 发现 {len(candidate_regions)} 个候选区域")

        # ═══════════════════════════════════════════════════════════
        # Stage 2: 精扫描 — 仅对候选区域逐秒分析
        # ═══════════════════════════════════════════════════════════
        text_frames = []
        total_candidates = len(candidate_regions)
        
        for idx, (region_start, region_end) in enumerate(candidate_regions):
            if progress_callback:
                coarse_progress = 0.4  # Stage1 占 40%
                fine_weight = 0.5      # Stage2 占 50%
                p = coarse_progress + fine_weight * ((idx + 1) / max(1, total_candidates))
                progress_callback(p, f"精扫候选区 {idx+1}/{total_candidates}: {region_start:.0f}s-{region_end:.0f}s")
            
            frames_in_region = self._fine_scan_region(
                cap, fps, total_frames, region_start, region_end, ocr
            )
            text_frames.extend(frames_in_region)

        cap.release()
        
        logger.info(f"[完成] 检测到 {len(text_frames)} 个静态文字画面")
        self.last_text_frames = text_frames
        return text_frames

    def _read_frame_at(self, cap, fps, total_frames, sec):
        """
        在指定秒数处读取一帧，返回 (success, frame, gray, gray_small, brightness)
        如果帧过暗或过曝，返回 (True, None, ...) 表示应跳过
        """
        target_frame = min(int(sec * fps), total_frames - 1)
        target_frame = max(0, target_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            return False, None, None, None, 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        
        # 快速跳过暗帧/过曝帧（不太可能包含标题文字）
        if brightness < BRIGHTNESS_MIN or brightness > BRIGHTNESS_MAX:
            return True, None, None, None, brightness
        
        gray_small = cv2.resize(gray, (320, 180))
        return True, frame, gray, gray_small, brightness

    def _coarse_scan(self, cap, fps, total_frames, total_duration, int_duration, progress_callback=None):
        """
        Stage 1: 粗扫描，每 COARSE_SCAN_INTERVAL 秒采一帧
        找出可能是静态画面的候选时间区域
        
        Returns:
            [(start_sec, end_sec), ...] 候选区域列表
        """
        prev_gray = None
        static_start = None
        static_counter = 0
        candidate_regions = []
        
        # 粗扫采样间隔（秒）
        interval = COARSE_SCAN_INTERVAL
        coarse_frames = max(1, int(interval * fps))
        
        for sec in range(0, int_duration, interval):
            ret, frame, gray, gray_small, brightness = self._read_frame_at(
                cap, fps, total_frames, sec
            )
            if not ret:
                break
            if frame is None:  # 跳过的暗/亮帧
                prev_gray = None
                static_start = None
                continue
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray_small, prev_gray)
                change_score = float(np.mean(diff))

                if change_score < STATIC_THRESHOLD:
                    if static_start is None:
                        static_start = sec
                    static_counter = 0
                else:
                    static_counter += 1
                    # 粗扫阶段容忍度稍大（因为采样稀疏）
                    if static_counter <= MAX_STATIC_GAP * 2:
                        continue
                    if static_start is not None:
                        duration = sec - static_start - MAX_STATIC_GAP * 2
                        if duration >= MIN_STATIC_DURATION:
                            # 记录候选区域（扩展边界确保覆盖完整）
                            exp_start = max(0, static_start - CANDIDATE_EXPAND_SEC)
                            exp_end = min(total_duration, sec + CANDIDATE_EXPAND_SEC)
                            candidate_regions.append((exp_start, exp_end))
                    static_start = None
            
            prev_gray = gray_small
            
            if progress_callback and sec % (10 * interval) == 0:
                progress_callback(
                    min(sec / total_duration, 0.39) * 0.4,
                    f"粗扫: {sec}/{int_duration}s"
                )

        # 处理末尾仍在静态的情况
        if static_start is not None:
            duration = int_duration - static_start
            if duration >= MIN_STATIC_DURATION:
                exp_start = max(0, static_start - CANDIDATE_EXPAND_SEC)
                candidate_regions.append((exp_start, total_duration))

        # 合并重叠/相邻的候选区域
        candidate_regions = self._merge_regions(candidate_regions, gap=10)
        
        return candidate_regions

    def _fine_scan_region(self, cap, fps, total_frames, region_start, region_end, ocr):
        """
        Stage 2: 对单个候选区域进行精细逐秒扫描
        确认静态段、执行 OCR、计算得分
        
        Returns:
            [text_frame_dict, ...] 该区域内检测到的文字帧
        """
        text_frames = []
        prev_gray = None
        static_start = None
        static_counter = 0
        frame_interval = max(1, int(fps))
        
        start_int = int(region_start)
        end_int = int(region_end)

        for sec in range(start_int, end_int):
            ret, frame, gray, gray_small, brightness = self._read_frame_at(
                cap, fps, total_frames, sec
            )
            if not ret:
                break
            if frame is None:
                # 暗帧/亮帧：如果之前在静态段中，算作可容忍的间隙
                if static_start is not None:
                    static_counter += 1
                    if static_counter > MAX_STATIC_GAP * 2:
                        # 结束当前静态段
                        duration = sec - static_start - MAX_STATIC_GAP
                        if duration >= MIN_STATIC_DURATION:
                            text_frames.append(self._analyze_static_segment(
                                cap, fps, total_frames, static_start, 
                                duration, ocr
                            ))
                        static_start = None
                        static_counter = 0
                continue

            if prev_gray is not None:
                diff = cv2.absdiff(gray_small, prev_gray)
                change_score = float(np.mean(diff))

                if change_score < STATIC_THRESHOLD:
                    if static_start is None:
                        static_start = sec
                    static_counter = 0  # 重置
                else:
                    static_counter += 1
                    if static_counter <= MAX_STATIC_GAP:
                        continue
                    # 结束当前静态段
                    if static_start is not None:
                        duration = sec - static_start - MAX_STATIC_GAP
                        if duration >= MIN_STATIC_DURATION:
                            text_frames.append(self._analyze_static_segment(
                                cap, fps, total_frames, static_start, 
                                duration, ocr
                            ))
                        static_start = None
            
            prev_gray = gray_small

        # 处理区域末尾仍在静态段的情况
        if static_start is not None:
            duration = end_int - static_start
            if duration >= MIN_STATIC_DURATION:
                text_frames.append(self._analyze_static_segment(
                    cap, fps, total_frames, static_start, duration, ocr
                ))

        return text_frames

    def _analyze_static_segment(self, cap, fps, total_frames, static_start, duration, ocr):
        """
        对确认的静态段执行 OCR 分析和评分
        
        Returns:
            文字帧字典
        """
        mid_sec = static_start + duration // 2
        ret, frame, gray, _, brightness = self._read_frame_at(cap, fps, total_frames, mid_sec)
        
        if not ret or frame is None:
            # 无法读取中点帧，用纯视觉特征返回低分结果
            return {
                'time': static_start,
                'duration': duration,
                'score': min(1.0, duration / 5.0) * 0.5,
                'type': 'unknown',
                'brightness': brightness,
                'edge_density': 0,
                'text': '',
                'ocr_score': 0,
            }
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size

        # OCR 识别（使用公共函数，自动降分辨率）
        ocr_text, ocr_score, has_chinese = _run_ocr(ocr, frame)

        # 综合评分
        base_score = min(1.0, duration / 5.0) * (1 + edge_density * 3)
        if ocr_text and len(ocr_text) > 2:
            text_bonus = 0.5 + ocr_score * 0.5
            if has_chinese:
                text_bonus += 0.8
            base_score += text_bonus

        return {
            'time': static_start,
            'duration': duration,
            'score': base_score,
            'type': 'title' if brightness > 80 else 'subtitle',
            'brightness': brightness,
            'edge_density': edge_density,
            'text': ocr_text[:100] if ocr_text else "",
            'ocr_score': ocr_score,
        }

    @staticmethod
    def _merge_regions(regions, gap=10):
        """
        合并重叠或相邻的区域
        
        Args:
            regions: [(start, end), ...]
            gap: 小于此间隔的区域会被合并
        
        Returns:
            合并后的区域列表
        """
        if not regions:
            return []
        
        sorted_regions = sorted(regions, key=lambda r: r[0])
        merged = [list(sorted_regions[0])]
        
        for start, end in sorted_regions[1:]:
            last_start, last_end = merged[-1]
            if start - last_end <= gap:
                # 合并
                merged[-1][1] = max(last_end, end)
            else:
                merged.append([start, end])
        
        return [(s, e) for s, e in merged]

    def _analyze_frame(self, frame, prev_gray) -> dict:
        """分析单帧是否包含文字"""
        # 转为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 计算边缘密度（文字通常有丰富的边缘）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. 计算亮度（文字画面通常较亮或较暗，但不是中间调）
        brightness = np.mean(gray) / 255.0
        
        # 3. 检测规则的文字区域（高对比度矩形区域）
        # 使用自适应阈值
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. 检测白色/高亮区域（可能是文字）
        white_ratio = np.sum(thresh > 200) / thresh.size
        black_ratio = np.sum(thresh < 50) / thresh.size
        
        # 5. 检测边缘方向（文字有水平/垂直边缘）
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 水平边缘占比
        h_edge = np.sum(np.abs(sobely) > 50) / sobely.size
        # 垂直边缘占比
        v_edge = np.sum(np.abs(sobelx) > 50) / sobelx.size
        
        # 6. 计算变化（与前一帧的差异）
        change_score = 0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            change_score = np.mean(diff) / 255.0

        # 综合评分 - 更严格的阈值
        score = 0.0
        is_text = False
        text_type = "unknown"

        # 文字画面特征组合：
        # - 边缘密度适中（不是纯文字也不是纯图像）
        # - 可能有大量白色或黑色区域
        # - 水平/垂直边缘明显
        
        if 0.03 < edge_density < 0.12:  # 边缘密度更严格
            score += 0.3
        
        if white_ratio > 0.25 or black_ratio > 0.35:  # 更严格的区域要求
            score += 0.3
        
        if h_edge > 0.015 or v_edge > 0.015:  # 更明显的边缘
            score += 0.25
        
        # 检测是否为标题画面（高亮背景+文字）
        if white_ratio > 0.4 and edge_density > 0.04:
            score += 0.35
            text_type = "title"
        
        # 检测是否为副标题（暗背景+文字）
        if black_ratio > 0.4 and edge_density > 0.04:
            score += 0.25
            text_type = "subtitle"

        if score > 0.7:  # 提高阈值
            is_text = True

        return {
            'is_text_frame': is_text,
            'score': score,
            'text_type': text_type,
            'brightness': brightness,
            'edge_density': edge_density,
            'white_ratio': white_ratio,
            'black_ratio': black_ratio,
            'change_score': change_score,
        }

    def find_title_boundaries(
        self,
        video_path: str,
        min_interval: float = 60.0,
        max_boundaries: int = 15,
        progress_callback=None
    ) -> List[float]:
        """
        找到标题画面出现的时间点作为歌曲边界

        Args:
            video_path: 视频路径
            min_interval: 最小间隔（秒）
            max_boundaries: 最多保留多少个边界

        Returns:
            标题画面的时间点列表
        """
        text_frames = self.detect_text_frames(video_path, progress_callback)

        if not text_frames:
            return []

        # 过滤低置信度（有OCR文字的优先，尤其是中文）
        text_frames = sorted(text_frames, key=lambda x: x['score'], reverse=True)

        # 输出识别结果
        logger.info("检测到的文字画面:")
        for f in text_frames[:20]:
            txt = f.get('text', '')[:30] if f.get('text') else ''
            logger.info(f"  {f['time']:.0f}s: score={f['score']:.2f} ocr={f.get('ocr_score',0):.2f} text={txt}")

        # 提取时间点（优先选择有 OCR 文字的，尤其是中文）
        boundaries = []
        for f in text_frames:
            ocr_text = f.get('text', '')
            has_ocr_content = len(ocr_text) > 2
            # 降低阈值：有OCR文字的 0.6 即可，纯视觉的 0.8
            threshold = 0.6 if has_ocr_content else 0.8
            if f['score'] > threshold:
                boundaries.append(f['time'])
        
        # 去重并合并相近的
        result = []
        last = -min_interval
        
        for t in sorted(boundaries):
            if t - last >= min_interval:
                result.append(float(t))
                last = t
        
        logger.info(f"标题边界: {len(result)}个 -> {result}")
        return result


if __name__ == "__main__":
    # 测试
    detector = TextFrameDetector()
    video = r"D:\个人资料\音乐测试\视频素材\live\蔡依林 - 2025 TMEA 腾讯音乐娱乐盛典表演部分 [牛牛视听馆]niutv.taobao.com.mp4"
    
    print("检测文字画面...")
    boundaries = detector.find_title_boundaries(video, min_interval=30)
    print(f"找到 {len(boundaries)} 个标题画面:")
    for t in boundaries:
        print(f"  {t:.1f}秒")
