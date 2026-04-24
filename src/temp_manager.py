"""
临时文件管理器

负责临时文件的创建、分类存储和自动清理
"""

import os
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TempFileManager:
    """临时文件管理器
    
    负责临时文件的创建、分类存储和自动清理
    """
    
    # 临时文件分类定义
    PREVIEW_DIR = "previews"
    UPLOAD_DIR = "uploads"
    CACHE_DIR = "cache"
    ASR_DIR = "asr_temp"
    
    # 过期时间（小时）
    PREVIEW_EXPIRE_HOURS = 24  # 预览文件24小时后过期
    UPLOAD_EXPIRE_HOURS = 48  # 上传文件48小时后过期
    
    def __init__(self, temp_root: str = None):
        if temp_root is None:
            project_root = Path(__file__).parent.parent
            self.temp_root = project_root / "temp"
        else:
            self.temp_root = Path(temp_root)
        
        # 初始化各分类目录
        self._init_dirs()
    
    def _init_dirs(self):
        """初始化临时文件目录结构"""
        dirs = [
            self.temp_root / self.PREVIEW_DIR,
            self.temp_root / self.UPLOAD_DIR,
            self.temp_root / self.CACHE_DIR,
            self.temp_root / self.ASR_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"临时文件目录已初始化: {self.temp_root}")
    
    def get_preview_path(self, filename: str) -> str:
        """获取预览文件存储路径"""
        return str(self.temp_root / self.PREVIEW_DIR / filename)
    
    def get_upload_path(self, filename: str) -> str:
        """获取上传文件存储路径"""
        return str(self.temp_root / self.UPLOAD_DIR / filename)
    
    def get_cache_path(self, filename: str) -> str:
        """获取缓存文件存储路径"""
        return str(self.temp_root / self.CACHE_DIR / filename)
    
    def get_asr_temp_path(self, filename: str) -> str:
        """获取ASR临时文件存储路径"""
        return str(self.temp_root / self.ASR_DIR / filename)
    
    def create_temp_file(self, suffix: str = None, prefix: str = "temp_", dir_type: str = CACHE_DIR) -> str:
        """创建临时文件，返回路径"""
        temp_dir = self.temp_root / dir_type
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(temp_dir))
        os.close(fd)
        return path
    
    def cleanup_old_files(self, older_than_hours: Optional[int] = None) -> int:
        """清理旧的临时文件
        
        Args:
            older_than_hours: 只清理早于指定小时数的文件，None表示用默认值
            
        Returns:
            删除的文件数
        """
        deleted_count = 0
        
        # 定义各目录的过期时间
        cleanup_rules = [
            (self.PREVIEW_DIR, older_than_hours or self.PREVIEW_EXPIRE_HOURS),
            (self.UPLOAD_DIR, older_than_hours or self.UPLOAD_EXPIRE_HOURS),
        ]
        
        for dir_name, expire_hours in cleanup_rules:
            dir_path = self.temp_root / dir_name
            if not dir_path.exists():
                continue
            
            cutoff_time = time.time() - (expire_hours * 3600)
            
            for file_path in dir_path.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"清理旧文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"清理文件失败 {file_path}: {e}")
        
        return deleted_count
    
    def cleanup_all_previews(self) -> int:
        """清理所有预览文件"""
        deleted_count = 0
        preview_dir = self.temp_root / self.PREVIEW_DIR
        if preview_dir.exists():
            for file_path in preview_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"清理预览文件失败 {file_path}: {e}")
        return deleted_count
    
    def cleanup_all_uploads(self) -> int:
        """清理所有上传文件"""
        deleted_count = 0
        upload_dir = self.temp_root / self.UPLOAD_DIR
        if upload_dir.exists():
            for file_path in upload_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"清理上传文件失败 {file_path}: {e}")
        return deleted_count
    
    def cleanup_all_temp(self) -> int:
        """清理所有临时文件"""
        deleted_count = 0
        total_count = 0
        
        for dir_name in [self.PREVIEW_DIR, self.UPLOAD_DIR, self.CACHE_DIR, self.ASR_DIR]:
            dir_path = self.temp_root / dir_name
            if not dir_path.exists():
                continue
            
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    total_count += 1
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"清理临时文件失败 {file_path}: {e}")
        
        return deleted_count
    
    def get_temp_size(self) -> Dict[str, float]:
        """获取临时文件总大小（MB）
        
        Returns:
            各分类目录的大小字典
        """
        sizes = {
            "previews": 0.0,
            "uploads": 0.0,
            "cache": 0.0,
            "asr_temp": 0.0,
            "total": 0.0,
        }
        
        for dir_name in [self.PREVIEW_DIR, self.UPLOAD_DIR, self.CACHE_DIR, self.ASR_DIR]:
            dir_path = self.temp_root / dir_name
            if not dir_path.exists():
                continue
            
            dir_size = 0
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    dir_size += file_path.stat().st_size
            
            size_mb = round(dir_size / (1024 * 1024), 2)
            if dir_name == self.PREVIEW_DIR:
                sizes["previews"] = size_mb
            elif dir_name == self.UPLOAD_DIR:
                sizes["uploads"] = size_mb
            elif dir_name == self.CACHE_DIR:
                sizes["cache"] = size_mb
            elif dir_name == self.ASR_DIR:
                sizes["asr_temp"] = size_mb
            
            sizes["total"] += size_mb
        
        sizes["total"] = round(sizes["total"], 2)
        return sizes
    
    def get_preview_count(self) -> int:
        """获取预览文件数量"""
        count = 0
        preview_dir = self.temp_root / self.PREVIEW_DIR
        if preview_dir.exists():
            count = len(list(preview_dir.glob("*")))
        return count
    
    def get_upload_count(self) -> int:
        """获取上传文件数量"""
        count = 0
        upload_dir = self.temp_root / self.UPLOAD_DIR
        if upload_dir.exists():
            count = len(list(upload_dir.glob("*")))
        return count
    
    def cleanup_legacy_temp_files(self, output_dir: str = None) -> Dict[str, int]:
        """清理历史遗留的临时文件（output 目录下的 temp_* 和 upload_*）
        
        Args:
            output_dir: 输出目录，默认为项目根目录下的 output 文件夹
            
        Returns:
            删除文件的统计信息字典
        """
        if output_dir is None:
            output_dir = self.temp_root.parent / "output"
        
        output_path = Path(output_dir)
        if not output_path.exists():
            return {"total": 0, "size_mb": 0}
        
        deleted_count = 0
        total_size = 0
        
        patterns = ["temp_preview_*", "temp_subtitle_*", "upload_*"]
        
        for pattern in patterns:
            for file_path in output_path.glob(pattern):
                if file_path.is_file():
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        total_size += file_size
                        logger.debug(f"清理历史临时文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"清理历史临时文件失败 {file_path}: {e}")
        
        return {
            "total": deleted_count,
            "size_mb": round(total_size / (1024 * 1024), 2)
        }


# 全局实例
_temp_manager: Optional[TempFileManager] = None


def get_temp_manager() -> TempFileManager:
    """获取全局临时文件管理器实例"""
    global _temp_manager
    if _temp_manager is None:
        _temp_manager = TempFileManager()
    return _temp_manager
