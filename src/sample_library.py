"""
样本库管理模块
"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.data_models import VideoSample, OptimizationHistory

logger = logging.getLogger(__name__)


class SampleLibraryDatabase:
    """
    样本库数据库管理
    """
    
    def __init__(self, db_path: str = "data/sample_library.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
    
    def _init_db(self):
        """
        初始化数据库表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 样本表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    video_duration REAL NOT NULL,
                    segments_json TEXT NOT NULL,
                    original_result_json TEXT,
                    config_version TEXT,
                    model_version TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    created_by TEXT,
                    tags_json TEXT,
                    notes TEXT
                )
            ''')
            
            # 优化历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_history (
                    optimization_id TEXT PRIMARY KEY,
                    before_config_json TEXT NOT NULL,
                    after_config_json TEXT NOT NULL,
                    before_accuracy_json TEXT NOT NULL,
                    after_accuracy_json TEXT NOT NULL,
                    applied INTEGER NOT NULL DEFAULT 0,
                    applied_at TEXT,
                    created_at TEXT NOT NULL,
                    notes TEXT
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_samples_created ON samples (created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_optimization_created ON optimization_history (created_at)')
            
            conn.commit()
    
    # ========== 样本管理 ==========
    
    def add_sample(self, sample: VideoSample) -> bool:
        """
        添加样本
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO samples
                    (sample_id, video_path, video_duration, segments_json, original_result_json,
                     config_version, model_version, created_at, last_updated, created_by,
                     tags_json, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sample.sample_id,
                    sample.video_path,
                    sample.video_duration,
                    json.dumps([seg.to_dict() for seg in sample.segments]),
                    json.dumps(sample._serialize_original_result()),
                    sample.config_version,
                    sample.model_version,
                    sample.created_at.isoformat(),
                    sample.last_updated.isoformat(),
                    sample.created_by,
                    json.dumps(sample.tags),
                    sample.notes,
                ))
                conn.commit()
                logger.info(f"样本已添加: {sample.sample_id}")
                return True
        except Exception as e:
            logger.error(f"添加样本失败: {e}")
            return False
    
    def update_sample(self, sample: VideoSample) -> bool:
        """
        更新样本
        """
        try:
            sample.last_updated = datetime.now()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE samples
                    SET segments_json = ?, original_result_json = ?,
                        config_version = ?, model_version = ?,
                        last_updated = ?, tags_json = ?, notes = ?
                    WHERE sample_id = ?
                ''', (
                    json.dumps([seg.to_dict() for seg in sample.segments]),
                    json.dumps(sample._serialize_original_result()),
                    sample.config_version,
                    sample.model_version,
                    sample.last_updated.isoformat(),
                    json.dumps(sample.tags),
                    sample.notes,
                    sample.sample_id,
                ))
                conn.commit()
                logger.info(f"样本已更新: {sample.sample_id}")
                return True
        except Exception as e:
            logger.error(f"更新样本失败: {e}")
            return False
    
    def get_sample(self, sample_id: str) -> Optional[VideoSample]:
        """
        获取单个样本
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM samples WHERE sample_id = ?', (sample_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_sample(row)
                return None
        except Exception as e:
            logger.error(f"获取样本失败: {e}")
            return None
    
    def get_all_samples(self, limit: int = 100) -> List[VideoSample]:
        """
        获取所有样本
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM samples ORDER BY created_at DESC LIMIT ?', (limit,))
                rows = cursor.fetchall()
                return [self._row_to_sample(row) for row in rows]
        except Exception as e:
            logger.error(f"获取样本列表失败: {e}")
            return []
    
    def delete_sample(self, sample_id: str) -> bool:
        """
        删除样本
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM samples WHERE sample_id = ?', (sample_id,))
                conn.commit()
                logger.info(f"样本已删除: {sample_id}")
                return True
        except Exception as e:
            logger.error(f"删除样本失败: {e}")
            return False
    
    def get_sample_count(self) -> int:
        """
        获取样本数量
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM samples')
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"获取样本数量失败: {e}")
            return 0
    
    def _row_to_sample(self, row) -> VideoSample:
        """
        将数据库行转换为VideoSample对象
        """
        from src.data_models import EditableSegment
        # 解析segments_json
        segments = []
        if row[3]:
            segments_data = json.loads(row[3])
            for seg_data in segments_data:
                segments.append(EditableSegment.from_dict(seg_data))
        
        # 解析tags_json
        tags = []
        if row[10]:
            tags = json.loads(row[10])
        
        # 构造样本（original_result需要特殊处理）
        from src.data_models import VideoSample
        from src.audio_analyzer import AnalysisResult
        sample = VideoSample(
            sample_id=row[0],
            video_path=row[1],
            video_duration=row[2],
            segments=segments,
            original_result=AnalysisResult([], '', 0.0, {}, 0.0),  # 占位
            config_version=row[5],
            model_version=row[6],
            created_at=datetime.fromisoformat(row[7]),
            last_updated=datetime.fromisoformat(row[8]),
            created_by=row[9],
            tags=tags,
            notes=row[11],
        )
        return sample
    
    # ========== 优化历史管理 ==========
    
    def add_optimization_history(self, history: OptimizationHistory) -> bool:
        """
        添加优化历史
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO optimization_history
                    (optimization_id, before_config_json, after_config_json,
                     before_accuracy_json, after_accuracy_json,
                     applied, applied_at, created_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    history.optimization_id,
                    json.dumps(history.before_config),
                    json.dumps(history.after_config),
                    json.dumps(history.before_accuracy),
                    json.dumps(history.after_accuracy),
                    1 if history.applied else 0,
                    history.applied_at.isoformat() if history.applied_at else None,
                    history.created_at.isoformat(),
                    history.notes,
                ))
                conn.commit()
                logger.info(f"优化历史已添加: {history.optimization_id}")
                return True
        except Exception as e:
            logger.error(f"添加优化历史失败: {e}")
            return False
    
    def update_optimization_applied(self, optimization_id: str, applied: bool) -> bool:
        """
        更新优化历史的应用状态
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE optimization_history
                    SET applied = ?, applied_at = ?
                    WHERE optimization_id = ?
                ''', (
                    1 if applied else 0,
                    datetime.now().isoformat() if applied else None,
                    optimization_id,
                ))
                conn.commit()
                logger.info(f"优化历史已更新: {optimization_id}, applied={applied}")
                return True
        except Exception as e:
            logger.error(f"更新优化历史失败: {e}")
            return False
    
    def get_optimization_history(self, limit: int = 20) -> List[OptimizationHistory]:
        """
        获取优化历史
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM optimization_history
                    ORDER BY created_at DESC LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
                
                histories = []
                for row in rows:
                    history = OptimizationHistory(
                        optimization_id=row[0],
                        before_config=json.loads(row[1]),
                        after_config=json.loads(row[2]),
                        before_accuracy=json.loads(row[3]),
                        after_accuracy=json.loads(row[4]),
                        applied=bool(row[5]),
                        applied_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        created_at=datetime.fromisoformat(row[7]),
                        notes=row[8],
                    )
                    histories.append(history)
                return histories
        except Exception as e:
            logger.error(f"获取优化历史失败: {e}")
            return []
    
    # ========== 工具方法 ==========
    
    def generate_sample_id(self) -> str:
        """
        生成样本ID
        """
        return f"sample_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def generate_optimization_id(self) -> str:
        """
        生成优化ID
        """
        return f"opt_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# 全局实例
_sample_library_db: Optional[SampleLibraryDatabase] = None


def get_sample_library_db() -> SampleLibraryDatabase:
    """
    获取样本库数据库实例
    """
    global _sample_library_db
    if _sample_library_db is None:
        _sample_library_db = SampleLibraryDatabase()
    return _sample_library_db

