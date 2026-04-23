"""
视频片段编辑与管理系统 - 统一数据模型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path

# 导入现有数据结构
from src.audio_analyzer import AnalysisResult


@dataclass
class EditableSegment:
    """
    可编辑的片段数据模型
    """
    segment_id: str
    start_time: float
    end_time: float
    original_label: str
    current_label: str
    confidence: float
    is_modified: bool = False
    original_confidence: float = field(init=False)
    
    def __post_init__(self):
        self.original_confidence = self.confidence
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment_id': self.segment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'original_label': self.original_label,
            'current_label': self.current_label,
            'confidence': self.confidence,
            'is_modified': self.is_modified,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EditableSegment':
        return cls(
            segment_id=data['segment_id'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            original_label=data['original_label'],
            current_label=data['current_label'],
            confidence=data['confidence'],
            is_modified=data.get('is_modified', False),
        )


@dataclass
class VideoSample:
    """
    视频样本数据模型
    """
    sample_id: str
    video_path: str
    video_duration: float
    segments: List[EditableSegment]
    original_result: AnalysisResult
    config_version: str
    model_version: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    created_by: str = "user"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    @property
    def modified_segment_count(self) -> int:
        return sum(1 for seg in self.segments if seg.is_modified)
    
    @property
    def modification_rate(self) -> float:
        if not self.segments:
            return 0.0
        return self.modified_segment_count / len(self.segments)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': self.sample_id,
            'video_path': self.video_path,
            'video_duration': self.video_duration,
            'segments': [seg.to_dict() for seg in self.segments],
            'original_result': self._serialize_original_result(),
            'config_version': self.config_version,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'created_by': self.created_by,
            'tags': self.tags,
            'notes': self.notes,
        }
    
    def _serialize_original_result(self) -> Dict[str, Any]:
        # 简单序列化 AnalysisResult
        return {
            'total_duration': self.original_result.total_duration,
            'singer': getattr(self.original_result, 'singer', ''),
            'audio_info': getattr(self.original_result, 'audio_info', {}),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoSample':
        return cls(
            sample_id=data['sample_id'],
            video_path=data['video_path'],
            video_duration=data['video_duration'],
            segments=[EditableSegment.from_dict(seg) for seg in data['segments']],
            original_result=cls._deserialize_original_result(data['original_result']),
            config_version=data['config_version'],
            model_version=data['model_version'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            created_by=data.get('created_by', 'user'),
            tags=data.get('tags', []),
            notes=data.get('notes', ''),
        )
    
    @classmethod
    def _deserialize_original_result(cls, data: Dict[str, Any]) -> AnalysisResult:
        # 构造一个简单的 AnalysisResult（需要根据实际结构调整）
        from src.audio_analyzer import AnalysisResult, SongInfo
        # 临时构造，实际使用时需要完整重建
        result = AnalysisResult(
            songs=[],
            singer=data.get('singer', ''),
            total_duration=data.get('total_duration', 0.0),
            audio_info=data.get('audio_info', {}),
            analysis_time=0.0,
        )
        return result
    
    def save_to_file(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'VideoSample':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class OptimizationHistory:
    """
    优化历史记录
    """
    optimization_id: str
    before_config: Dict[str, Any]
    after_config: Dict[str, Any]
    before_accuracy: Dict[str, float]
    after_accuracy: Dict[str, float]
    applied: bool = False
    applied_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimization_id': self.optimization_id,
            'before_config': self.before_config,
            'after_config': self.after_config,
            'before_accuracy': self.before_accuracy,
            'after_accuracy': self.after_accuracy,
            'applied': self.applied,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'created_at': self.created_at.isoformat(),
            'notes': self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationHistory':
        return cls(
            optimization_id=data['optimization_id'],
            before_config=data['before_config'],
            after_config=data['after_config'],
            before_accuracy=data['before_accuracy'],
            after_accuracy=data['after_accuracy'],
            applied=data.get('applied', False),
            applied_at=datetime.fromisoformat(data['applied_at']) if data.get('applied_at') else None,
            created_at=datetime.fromisoformat(data['created_at']),
            notes=data.get('notes', ''),
        )
    
    def save_to_file(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'OptimizationHistory':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

