"""
预览编辑器模块
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.data_models import EditableSegment, VideoSample
from src.audio_analyzer import AnalysisResult
from src.sample_library import get_sample_library_db
from src.config import ProcessingConfig

logger = logging.getLogger(__name__)


class PreviewEditor:
    """
    预览编辑器主类
    """
    
    def __init__(self):
        self.db = get_sample_library_db()
        self.current_analysis_result: Optional[AnalysisResult] = None
        self.current_segments: List[EditableSegment] = []
        self.video_path: str = ""
        self.video_duration: float = 0.0
        self.config: Optional[ProcessingConfig] = None
    
    def load_video(self, analysis_result: AnalysisResult, video_path: str, config: ProcessingConfig):
        """
        加载视频分析结果到编辑器
        """
        self.current_analysis_result = analysis_result
        self.video_path = video_path
        self.video_duration = analysis_result.total_duration
        self.config = config
        
        # 将分析结果转换为可编辑的片段
        self._convert_analysis_to_editable(analysis_result)
        
        logger.info(f"编辑器已加载视频: {video_path}")
        return True
    
    def _convert_analysis_to_editable(self, analysis_result: AnalysisResult):
        """
        将分析结果转换为可编辑片段
        """
        self.current_segments = []
        
        segment_idx = 0
        for song in analysis_result.songs:
            for segment in song.segments:
                # 创建可编辑片段
                editable = EditableSegment(
                    segment_id=f"seg_{uuid.uuid4().hex[:8]}_{segment_idx}",
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    original_label=segment.label,
                    current_label=segment.label,
                    confidence=segment.confidence,
                    is_modified=False,
                )
                self.current_segments.append(editable)
                segment_idx += 1
    
    def update_segment_label(self, segment_id: str, new_label: str) -> bool:
        """
        更新片段标签
        """
        for segment in self.current_segments:
            if segment.segment_id == segment_id:
                if segment.current_label != new_label:
                    segment.current_label = new_label
                    segment.is_modified = True
                    logger.info(f"片段标签已更新: {segment_id} {segment.original_label} → {new_label}")
                return True
        return False
    
    def update_segment_time(self, segment_id: str, new_start: float, new_end: float) -> bool:
        """
        更新片段时间
        """
        for segment in self.current_segments:
            if segment.segment_id == segment_id:
                changed = False
                if segment.start_time != new_start:
                    segment.start_time = new_start
                    changed = True
                if segment.end_time != new_end:
                    segment.end_time = new_end
                    changed = True
                if changed:
                    segment.is_modified = True
                    logger.info(f"片段时间已更新: {segment_id}")
                return True
        return False
    
    def reset_segment(self, segment_id: str) -> bool:
        """
        重置片段到原始状态
        """
        for segment in self.current_segments:
            if segment.segment_id == segment_id:
                segment.current_label = segment.original_label
                segment.is_modified = False
                logger.info(f"片段已重置: {segment_id}")
                return True
        return False
    
    def reset_all_segments(self):
        """
        重置所有片段
        """
        for segment in self.current_segments:
            segment.current_label = segment.original_label
            segment.is_modified = False
        logger.info("所有片段已重置")
    
    def get_modified_segments(self) -> List[EditableSegment]:
        """
        获取所有已修改的片段
        """
        return [seg for seg in self.current_segments if seg.is_modified]
    
    def get_segments_for_export(self) -> List[EditableSegment]:
        """
        获取导出用的片段
        """
        return self.current_segments
    
    def save_as_sample(
        self,
        sample_name: str = "",
        notes: str = ""
    ) -> Optional[VideoSample]:
        """
        保存为样本
        """
        sample_id = self.db.generate_sample_id()
        
        # 创建样本对象
        sample = VideoSample(
            sample_id=sample_id,
            video_path=self.video_path,
            video_duration=self.video_duration,
            segments=self.current_segments,
            original_result=self.current_analysis_result,
            config_version="v1",  # 需要根据实际情况设置
            model_version="v1",    # 需要根据实际情况设置
            created_at=datetime.now(),
            last_updated=datetime.now(),
            created_by="user",
            notes=notes,
        )
        
        # 保存到数据库
        if self.db.add_sample(sample):
            logger.info(f"样本已保存: {sample_id}")
            return sample
        else:
            logger.error("保存样本失败")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取编辑器统计信息
        """
        total_segments = len(self.current_segments)
        modified_segments = len(self.get_modified_segments())
        label_counts = {}
        
        for segment in self.current_segments:
            label = segment.current_label
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            'total_segments': total_segments,
            'modified_segments': modified_segments,
            'modification_rate': modified_segments / total_segments if total_segments > 0 else 0.0,
            'label_distribution': label_counts,
        }


# 全局实例
_preview_editor: Optional[PreviewEditor] = None


def get_preview_editor() -> PreviewEditor:
    """
    获取预览编辑器实例
    """
    global _preview_editor
    if _preview_editor is None:
        _preview_editor = PreviewEditor()
    return _preview_editor

